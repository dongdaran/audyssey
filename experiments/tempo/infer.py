"""Single-file TEMPO caption inference in an isolated upstream environment."""
import argparse
import json
import sys
from pathlib import Path


# Table 1 questions, EMNLP 2026 TEMPO, p. 15962.
# https://aclanthology.org/2026.emnlp-main.694.pdf
# Appendix D / Table 7 suffixes are for zero-shot baselines, not TEMPO.
PROMPTS = {
    'caption': 'Which sound events occur, and what are their time intervals and descriptions?',
    'diar': 'How many speakers are in this audio and when does each one speak? Include timestamps.',
}


def restore_projector(model, checkpoint, swap, bind_safe):
    """Restore the projector on the module actually used by forward()."""
    owner = model if hasattr(model, 'multi_modal_projector') else model.model
    if not hasattr(owner, 'multi_modal_projector'):
        raise RuntimeError('Unsupported AudioFlamingo3 projector layout')
    swap(owner, time_dim=64, frame_duration=0.04, checkpoint_dir=str(checkpoint))
    bind_safe(owner)


def load_model(checkpoint):
    import torch
    from transformers import AutoProcessor, AudioFlamingo3ForConditionalGeneration
    from time_aware_projector import swap_to_time_aware_projector
    from infer_worker import _bind_safe_projector_forward

    processor = AutoProcessor.from_pretrained(str(checkpoint))
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        str(checkpoint), torch_dtype=torch.bfloat16)
    if len(processor.tokenizer) != model.get_input_embeddings().num_embeddings:
        raise RuntimeError('TEMPO tokenizer and model embedding sizes differ')
    restore_projector(model, checkpoint, swap_to_time_aware_projector,
                      _bind_safe_projector_forward)
    model.to('cuda:0').eval()
    model.config.use_cache = True
    return model, processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('audio', type=Path)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--model', default='Kaousheik/tempo-rl-multitask')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', choices=['caption', 'diar'], default='caption')
    args = parser.parse_args()
    import torch
    from huggingface_hub import snapshot_download
    if not torch.cuda.is_available():
        raise RuntimeError('TEMPO inference requires a CUDA GPU')
    sys.path[:0] = [str(args.repo.resolve() / 'evaluation'), str(args.repo.resolve() / 'sft')]
    from infer_worker import build_conversation, _NON_TS_SPECIALS
    checkpoint = Path(args.model)
    if not checkpoint.is_dir():
        checkpoint = Path(snapshot_download(args.model))
    if not (checkpoint / 'time_proj.pt').is_file():
        raise RuntimeError('TEMPO checkpoint is missing required time_proj.pt')
    model, processor = load_model(checkpoint)
    question = PROMPTS[args.task]
    conversation = build_conversation(question, str(args.audio.resolve()), None)
    inputs = processor.apply_chat_template([conversation], tokenize=True,
        add_generation_prompt=True, return_dict=True, output_labels=False)
    inputs = {k: value.to('cuda:0', dtype=torch.bfloat16) if value.is_floating_point()
              else value.to('cuda:0') for k, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=768, do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id)
    raw = processor.batch_decode(output[:, inputs['input_ids'].shape[1]:], skip_special_tokens=False)[0]
    for token in _NON_TS_SPECIALS:
        raw = raw.replace(token, '')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'model': args.model, 'task': args.task,
                                      'prompt': question, 'raw': raw.strip()}, indent=2))


if __name__ == '__main__':
    main()
