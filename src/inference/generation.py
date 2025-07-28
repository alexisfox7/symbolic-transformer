#./inference/generation.py
"""
Text Generation Utilities
"""

import torch
from accelerate.logging import get_logger
from tqdm.auto import tqdm
from typing import Tuple, List, Dict, Optional, Any

from mytokenizers import BaseTokenizer
from inference.hooks import InferenceHookManager, InferenceHook

logger = get_logger(__name__)

@torch.no_grad()
def run_generation(model: torch.nn.Module, 
                  tokenizer: BaseTokenizer,
                  prompt_text: str,
                  device: torch.device,
                  max_new_tokens: int = 50,
                  temperature: float = 1.0,
                  top_k: Optional[int] = None,
                  hooks: Optional[List[InferenceHook]] = None) -> Tuple[List[int], str]:
    """
    Generate text using the model starting from a prompt.

    Returns:
        Tuple of (list of token IDs, generated text string)
    """
    # ensure the model has a generate method
    if not hasattr(model, 'generate'):
        logger.error("Model does not have a 'generate' method required for this function.")
        raise AttributeError("Model must have a 'generate' method for text generation")

    # set the model to evaluation mode
    model.eval()
    model.to(device)
    
    # set up hooks if provided
    hook_manager = None
    if hooks and hasattr(model, 'set_hook_manager'):
        hook_manager = InferenceHookManager()
        for hook in hooks:
            hook_manager.add_hook(hook)
        model.set_hook_manager(hook_manager)
        logger.info(f"Attached {len(hooks)} inference hooks: {[h.name for h in hooks]}")

    logger.info(f"Generating text with parameters:")
    logger.info(f"  Prompt: '{prompt_text}'")
    logger.info(f"  Max new tokens: {max_new_tokens}")
    logger.info(f"  Temperature: {temperature}")
    logger.info(f"  Top-k: {top_k if top_k is not None else 'Not Used'}")

    # encode the starting prompt
    try:
        # add special tokens if needed (though model.generate usually handles context without BOS for prompt)
        start_ids = tokenizer.encode(prompt_text, add_special_tokens=True, return_tensors='pt')
        
        if not isinstance(start_ids, torch.Tensor):
            start_ids = torch.tensor([start_ids], dtype=torch.long)
            
        start_ids = start_ids.to(device)
        
        if start_ids.shape[1] == 0:
            logger.warning("Encoded prompt is empty. Using BOS token as fallback.")
            # ensure bos_token_id is available and is an int
            start_token_id = tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None else 0
            start_ids = torch.tensor([[start_token_id]], dtype=torch.long, device=device)
            
        logger.info(f"Encoded prompt IDs: {start_ids.tolist()}")
        
    except Exception as e:
        logger.error(f"Error encoding prompt: {e}", exc_info=True)
        raise


    progress_bar = tqdm(total=max_new_tokens, desc="Generating tokens")
    
    # prepare arguments for model.generate
    generate_kwargs = {
        'input_ids': start_ids,
        'max_new_tokens': max_new_tokens,
        'temperature': temperature,
        'tokenizer': tokenizer,  # pass tokenizer for hooks
    }
    if top_k is not None:
        generate_kwargs['top_k'] = top_k

    try:
        generated_ids_tensor = model.generate(**generate_kwargs)
        
        if isinstance(generated_ids_tensor, torch.Tensor):
            generated_ids = generated_ids_tensor[0].tolist()  
        else:
            generated_ids = generated_ids_tensor #type: ignore
            
        progress_bar.update(max_new_tokens)  
        progress_bar.close()
            
    except Exception as e:
        if 'progress_bar' in locals() and progress_bar: # check if progress_bar exists and is not None
            progress_bar.close()
        logger.error(f"Error during model.generate(): {e}", exc_info=True)
        raise

    try:
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Error decoding generated IDs: {e}")
        generated_text = "[Decoding Error]"

    logger.info("Generation complete")
    logger.info(f"Generated text:\n---\n{generated_text}\n---")
    
    # clean up hooks
    if hook_manager and hasattr(model, 'set_hook_manager'):
        model.set_hook_manager(None)
        logger.info("Removed inference hooks")

    return generated_ids, generated_text


def batch_generate(model: torch.nn.Module, 
                  tokenizer: BaseTokenizer,
                  prompts: List[str],
                  device: torch.device,
                  hooks: Optional[List[InferenceHook]] = None,
                  **kwargs) -> List[Tuple[List[int], str]]:
    """
    Generate text for multiple prompts.
    
    Args:
        prompts: List of prompt texts
        **kwargs: Additional arguments for generation (max_new_tokens, temperature, top_k, top_p)
        
    Returns:
        List of (token IDs, generated text) tuples
    """
    results = []

    for i, prompt in enumerate(prompts):
        logger.info(f"\nGenerating text for prompt {i+1}/{len(prompts)}: '{prompt}'")
        
        try:
            result = run_generation(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt,
                device=device,
                hooks=hooks,
                **kwargs #pass along all kwargs (including top_p if present)
            )
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error generating for prompt {i+1}: {e}")
            results.append(([], f"[Generation Error: {str(e)}]"))
    
    return results

