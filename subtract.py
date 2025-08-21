import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import matplotlib.pyplot as plt

class InteractiveTokenSandbox:
    """
    Interactive sandbox for subtracting ANY token from residuals.
    """
    
    def __init__(self, model_name='gpt2', device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model.eval()
        
        # Current state
        self.current_residual = None
        self.original_residual = None
        self.subtraction_history = []
        
        print(f"Loaded model on {self.device}")
        print(f"Vocabulary size: {self.tokenizer.vocab_size}")
        
    def set_target(self, text: str, layer: int, position: int):
        """Set the target residual to work with."""
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            self.original_residual = outputs.hidden_states[layer][0, position, :].clone()
            self.current_residual = self.original_residual.clone()
            
        self.text = text
        self.layer = layer
        self.position = position
        self.subtraction_history = []
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        print(f"\n{'='*80}")
        print(f"Target set:")
        print(f"  Text: {text}")
        print(f"  Tokens: {tokens}")
        print(f"  Layer: {layer}, Position: {position} ('{tokens[position]}')")
        print(f"  Original norm: {torch.norm(self.original_residual).item():.3f}")
        print(f"{'='*80}")
        
    def search_tokens(self, query: str, limit: int = 10):
        """Search for tokens containing a string."""
        results = []
        for token_id in range(self.tokenizer.vocab_size):
            token_str = self.tokenizer.decode([token_id])
            if query.lower() in token_str.lower():
                results.append((token_id, token_str))
                if len(results) >= limit:
                    break
        
        print(f"\nTokens containing '{query}':")
        for token_id, token_str in results:
            print(f"  {token_id:5d}: '{token_str}'")
        
        return results
    
    def get_token_by_id(self, token_id: int) -> str:
        """Get token string by ID."""
        return self.tokenizer.decode([token_id])
    
    def get_token_id(self, token_str: str) -> int:
        """Get token ID from string."""
        # Try to encode without special tokens
        ids = self.tokenizer.encode(token_str, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
        else:
            print(f"Warning: '{token_str}' encodes to multiple tokens: {ids}")
            return ids[0] if ids else None
    
    def subtract(self, token_input, scale: float = None, method: str = 'first_layer'):
        """
        Subtract a token from current residual.
        
        Args:
            token_input: Can be:
                - str: Token string like " fox"
                - int: Token ID like 12345
                - 'random': Random token from vocabulary
                - 'top': Top token according to LogitLens
            scale: Scaling factor (None = optimize automatically)
            method: How to get embedding ('raw', 'normalized', 'scaled', 'first_layer')
        """
        if self.current_residual is None:
            print("Error: No target set. Use set_target() first.")
            return
        
        # Get token ID
        if isinstance(token_input, str):
            if token_input == 'random':
                token_id = np.random.randint(0, self.tokenizer.vocab_size)
                token_str = self.tokenizer.decode([token_id])
            elif token_input == 'top':
                token_id, token_str = self._get_top_token()
            else:
                token_id = self.get_token_id(token_input)
                token_str = token_input
        elif isinstance(token_input, int):
            token_id = token_input
            token_str = self.tokenizer.decode([token_id])
        else:
            print(f"Error: Invalid token input: {token_input}")
            return
        
        # Get embedding
        with torch.no_grad():
            if method == 'first_layer':
                token_embed = self._get_token_through_first_layer(token_id, self.position)
            else:
                token_embed = self.model.transformer.wte(torch.tensor([token_id], device=self.device))[0]
                
                if method == 'normalized':
                    token_embed = self.model.transformer.ln_f(token_embed.unsqueeze(0))[0]
                elif method == 'scaled':
                    # Scale to match residual norm
                    token_embed = token_embed * (torch.norm(self.current_residual) / torch.norm(token_embed))
        
        # Optimize scale if not provided
        if scale is None:
            scale = torch.dot(self.current_residual, token_embed) / torch.dot(token_embed, token_embed)
            scale = scale.item()
        
        # Store state before subtraction
        norm_before = torch.norm(self.current_residual).item()
        
        # Subtract
        self.current_residual = self.current_residual - scale * token_embed
        
        # Calculate metrics
        norm_after = torch.norm(self.current_residual).item()
        norm_reduction = norm_before - norm_after
        norm_reduction_pct = 100 * norm_reduction / norm_before if norm_before > 0 else 0
        
        cosine_sim = F.cosine_similarity(
            (self.original_residual - self.current_residual).unsqueeze(0),
            token_embed.unsqueeze(0)
        ).item()
        
        # Add to history
        self.subtraction_history.append({
            'token_id': token_id,
            'token_str': token_str,
            'scale': scale,
            'method': method,
            'norm_before': norm_before,
            'norm_after': norm_after,
            'reduction': norm_reduction,
            'reduction_pct': norm_reduction_pct,
            'cosine_sim': cosine_sim
        })
        
        # Print result
        print(f"\nSubtracted: '{token_str}' (ID: {token_id})")
        print(f"  Scale: {scale:.3f}")
        print(f"  Method: {method}")
        print(f"  Norm: {norm_before:.3f} → {norm_after:.3f}")
        print(f"  Reduction: {norm_reduction:.3f} ({norm_reduction_pct:.1f}%)")
        print(f"  Cosine similarity: {cosine_sim:.3f}")
        print(f"  Progress: {100 * (1 - norm_after / torch.norm(self.original_residual).item()):.1f}% total reduction")
        
    def _get_top_token(self):
        """Get top token according to LogitLens."""
        with torch.no_grad():
            residual_normed = self.model.transformer.ln_f(self.current_residual.unsqueeze(0))
            logits = self.model.lm_head(residual_normed)[0]
            token_id = torch.argmax(logits).item()
            token_str = self.tokenizer.decode([token_id])
        return token_id, token_str
    
    def _get_token_through_first_layer(self, token_id, position_idx):
        """
        Pass a single token through the first transformer layer at a specific position.
        Based on the approach from test1st.py.
        """
        # Get sequence length from current text
        input_ids = self.tokenizer.encode(self.text, return_tensors='pt').to(self.device)
        seq_length = input_ids.shape[1]
        
        # Create dummy input with just our token at the specified position
        pad_token_id = 50256
        dummy_input = torch.full((1, seq_length), pad_token_id, dtype=torch.long, device=self.device)
        dummy_input[0, position_idx] = token_id
        
        with torch.no_grad():
            # Get embeddings
            inputs_embeds = self.model.transformer.wte(dummy_input)
            position_ids = torch.arange(seq_length, device=self.device).unsqueeze(0)
            position_embeds = self.model.transformer.wpe(position_ids)
            
            # Initial hidden state (what goes into layer 0)
            hidden_states = inputs_embeds + position_embeds
            
            # Pass through first transformer block only
            first_block = self.model.transformer.h[0]
            
            # Create attention mask (attend only to the token position)
            attention_mask = torch.zeros((1, seq_length), device=self.device)
            attention_mask[0, position_idx] = 1.0
            
            # Extended attention mask for the transformer block
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            
            # Pass through first layer
            outputs = first_block(
                hidden_states,
                attention_mask=extended_attention_mask,
            )
            
            first_layer_output = outputs[0]
            
            return first_layer_output[0, position_idx, :]
    
    def undo(self, steps: int = 1):
        """Undo the last n subtractions."""
        if not self.subtraction_history:
            print("Nothing to undo")
            return
        
        # Reset to original
        self.current_residual = self.original_residual.clone()
        
        # Replay all but last n steps
        history_to_replay = self.subtraction_history[:-steps]
        self.subtraction_history = []
        
        for step in history_to_replay:
            self.subtract(step['token_id'], step['scale'], step['method'])
        
        print(f"\nUndone {steps} step(s)")
        
    def reset(self):
        """Reset to original residual."""
        self.current_residual = self.original_residual.clone()
        self.subtraction_history = []
        print("Reset to original residual")
        
    def status(self):
        """Show current status."""
        if self.current_residual is None:
            print("No target set")
            return
        
        original_norm = torch.norm(self.original_residual).item()
        current_norm = torch.norm(self.current_residual).item()
        
        print(f"\n{'='*60}")
        print(f"Current Status:")
        print(f"  Original norm: {original_norm:.3f}")
        print(f"  Current norm: {current_norm:.3f}")
        print(f"  Total reduction: {100 * (1 - current_norm / original_norm):.1f}%")
        print(f"  Steps taken: {len(self.subtraction_history)}")
        
        if self.subtraction_history:
            print(f"\nHistory:")
            for i, step in enumerate(self.subtraction_history):
                print(f"  {i+1}. '{step['token_str']}' × {step['scale']:.3f} "
                      f"(reduced {step['reduction_pct']:.1f}%)")
        
        # Show top 5 current predictions
        print(f"\nTop 5 predictions for current residual:")
        with torch.no_grad():
            residual_normed = self.model.transformer.ln_f(self.current_residual.unsqueeze(0))
            logits = self.model.lm_head(residual_normed)[0]
            top5_values, top5_indices = torch.topk(logits, k=5)
            
            for i, (value, idx) in enumerate(zip(top5_values, top5_indices)):
                token_str = self.tokenizer.decode([idx.item()])
                print(f"  {i+1}. '{token_str}' (logit: {value.item():.2f})")
        
        print(f"{'='*60}")
    
    def auto_subtract(self, n_steps: int = 5, method: str = 'first_layer'):
        """Automatically subtract top n tokens."""
        for i in range(n_steps):
            print(f"\nAuto step {i+1}/{n_steps}:")
            self.subtract('top', method=method)
    
    def experiment(self, tokens_to_try: list, method: str = 'first_layer'):
        """Try subtracting a list of tokens and compare."""
        results = []
        
        for token in tokens_to_try:
            # Reset to original
            self.current_residual = self.original_residual.clone()
            
            # Try subtracting this token
            self.subtract(token, method=method)
            
            # Store result
            if self.subtraction_history:
                results.append(self.subtraction_history[-1])
            
            # Reset history
            self.subtraction_history = []
        
        # Reset to original
        self.current_residual = self.original_residual.clone()
        
        # Sort by reduction
        results.sort(key=lambda x: x['reduction_pct'], reverse=True)
        
        print(f"\n{'='*60}")
        print(f"Experiment Results (method={method}):")
        print(f"{'='*60}")
        for i, result in enumerate(results[:10]):
            print(f"{i+1:2d}. '{result['token_str']:15s}' "
                  f"scale={result['scale']:7.3f} "
                  f"reduction={result['reduction_pct']:5.1f}% "
                  f"cos_sim={result['cosine_sim']:6.3f}")
        
        return results
    
    def plot_history(self):
        """Plot the subtraction history."""
        if not self.subtraction_history:
            print("No history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Norm over steps
        norms = [torch.norm(self.original_residual).item()]
        for step in self.subtraction_history:
            norms.append(step['norm_after'])
        
        axes[0, 0].plot(norms, marker='o')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Residual Norm')
        axes[0, 0].set_title('Norm Reduction Over Steps')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reduction per step
        reductions = [step['reduction_pct'] for step in self.subtraction_history]
        tokens = [step['token_str'][:10] for step in self.subtraction_history]
        
        axes[0, 1].bar(range(len(reductions)), reductions)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Reduction %')
        axes[0, 1].set_title('Reduction Per Step')
        axes[0, 1].set_xticks(range(len(tokens)))
        axes[0, 1].set_xticklabels(tokens, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Scales used
        scales = [step['scale'] for step in self.subtraction_history]
        axes[1, 0].bar(range(len(scales)), scales)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Scale Factor')
        axes[1, 0].set_title('Scale Factors Used')
        axes[1, 0].set_xticks(range(len(tokens)))
        axes[1, 0].set_xticklabels(tokens, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Cumulative reduction
        cumulative = []
        original_norm = torch.norm(self.original_residual).item()
        for step in self.subtraction_history:
            cumulative.append(100 * (1 - step['norm_after'] / original_norm))
        
        axes[1, 1].plot(cumulative, marker='o', color='green')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Total Reduction %')
        axes[1, 1].set_title('Cumulative Reduction')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=100, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()


def interactive_session():
    """
    Run an interactive session.
    """
    sandbox = InteractiveTokenSandbox()
    
    print("\n" + "="*80)
    print("INTERACTIVE TOKEN SUBTRACTION SANDBOX")
    print("="*80)
    print("\nCommands:")
    print("  set TEXT LAYER POS  - Set target residual")
    print("  sub TOKEN [SCALE]   - Subtract token (use quotes for spaces)")
    print("  sub random          - Subtract random token")
    print("  sub top             - Subtract top LogitLens token")
    print("  auto N              - Auto subtract top N tokens")
    print("  search QUERY        - Search for tokens containing string")
    print("  id TOKEN_ID         - Get token string from ID")
    print("  undo [N]            - Undo last N steps (default 1)")
    print("  reset               - Reset to original")
    print("  status              - Show current status")
    print("  plot                - Plot history")
    print("  exp TOKEN1 TOKEN2...  - Experiment with multiple tokens")
    print("  quit                - Exit")
    print("\nExample: set 'The cat sat on the mat' 5 2")
    print("         sub ' cat' 2.5")
    print("         sub random")
    
    # Default target for quick testing
    sandbox.set_target("The quick brown fox jumps over the lazy dog", 5, 3)
    
    while True:
        try:
            cmd = input("\n> ").strip()
            
            if cmd == 'quit':
                break
                
            elif cmd.startswith('set '):
                parts = cmd[4:].rsplit(' ', 2)
                if len(parts) == 3:
                    text = parts[0].strip("'\"")
                    layer = int(parts[1])
                    pos = int(parts[2])
                    sandbox.set_target(text, layer, pos)
                else:
                    print("Usage: set TEXT LAYER POS")
                    
            elif cmd.startswith('sub '):
                parts = cmd[4:].split(' ', 1)
                token = parts[0].strip("'\"")
                scale = float(parts[1]) if len(parts) > 1 else None
                sandbox.subtract(token, scale)
                
            elif cmd.startswith('auto'):
                n = int(cmd.split()[1]) if len(cmd.split()) > 1 else 5
                sandbox.auto_subtract(n)
                
            elif cmd.startswith('search '):
                query = cmd[7:]
                sandbox.search_tokens(query)
                
            elif cmd.startswith('id '):
                token_id = int(cmd[3:])
                token_str = sandbox.get_token_by_id(token_id)
                print(f"Token {token_id}: '{token_str}'")
                
            elif cmd == 'undo':
                sandbox.undo(1)
            elif cmd.startswith('undo '):
                n = int(cmd[5:])
                sandbox.undo(n)
                
            elif cmd == 'reset':
                sandbox.reset()
                
            elif cmd == 'status':
                sandbox.status()
                
            elif cmd == 'plot':
                sandbox.plot_history()
                
            elif cmd.startswith('exp '):
                tokens = cmd[4:].split()
                tokens = [t.strip("'\"") for t in tokens]
                sandbox.experiment(tokens)
                
            else:
                print(f"Unknown command: {cmd}")
                
        except KeyboardInterrupt:
            print("\nUse 'quit' to exit")
        except Exception as e:
            print(f"Error: {e}")
    
    print("Goodbye!")


if __name__ == "__main__":
    # Run interactive session
    #interactive_session()
    
    # Or use programmatically:
    sandbox = InteractiveTokenSandbox()
    sandbox.set_target("The cat sat on the mat", layer=5, position=2)
    sandbox.subtract(" quiet")
    #sandbox.subtract(" dog")
    #sandbox.subtract("random")
    sandbox.status()
    sandbox.plot_history()