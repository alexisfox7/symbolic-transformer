def evaluate_checkpoint(checkpoint_path, val_dataloader, base_config, device='cuda'):
    """Evaluate a single checkpoint on validation data."""
    print(f"🧪 Evaluating: {os.path.basename(checkpoint_path)}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Find model state dict
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', 0)
            global_batch = checkpoint.get('global_batch', 0)
            train_loss = checkpoint.get('loss', float('nan'))
        else:
            model_state_dict = checkpoint
            epoch = 0
            global_batch = 0
            train_loss = float('nan')
        
        # Simple fix: detect dimensions directly from weights
        import copy
        model_config = copy.deepcopy(base_config)
        
        # Clean up module prefix if present
        clean_state_dict = {}
        for key, value in model_state_dict.items():
            new_key = key.replace('module.', '') if key.startswith('module.') else key
            clean_state_dict[new_key] = value
        
        # Get dimensions from embedding layer
        if 'transformer.wte.weight' in clean_state_dict:
            vocab_size, d_model = clean_state_dict['transformer.wte.weight'].shape
            model_config.vocab_size = vocab_size
            model_config.d_model = d_model
            model_config.d_ff = 4 * d_model
            model_config.n_head = d_model // 64  # Standard 64-dim heads
            print(f"  Detected: d_model={d_model}, vocab_size={vocab_size}")
        
        # Count layers
        max_layer = 0
        for key in clean_state_dict.keys():
            if 'transformer.h.' in key:
                layer_num = int(key.split('transformer.h.')[1].split('.')[0])
                max_layer = max(max_layer, layer_num)
        model_config.n_layer = max_layer + 1
        
        # Create model (check if symbolic or vanilla)
        is_symbolic = any('symbolic' in key.lower() for key in clean_state_dict.keys()) or \
                     'symbolic' in checkpoint_path.lower()
        
        if is_symbolic:
            model = get_model("Symbolic", config=model_config).to(device)
        else:
            model = get_model("Vanilla", config=model_config).to(device)
        
        # Load weights
        model.load_state_dict(clean_state_dict)
        model.eval()
        
        # Run validation
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for val_batch_data in val_dataloader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in val_batch_data.items()}
                
                outputs = model(**batch)
                loss = outputs.get('loss')
                
                if loss is not None and not torch.isnan(loss):
                    batch_size = batch.get('input_ids', next(iter(batch.values()))).size(0)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
        
        # Calculate metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
        
        results = {
            'checkpoint': os.path.basename(checkpoint_path),
            'epoch': epoch,
            'global_batch': global_batch,
            'train_loss': train_loss,
            'val_loss': avg_loss,
            'val_perplexity': perplexity,
            'val_samples': total_samples
        }
        
        print(f"  Val Loss: {avg_loss:.4f}, Val PPL: {perplexity:.2f}")
        return results
        
    except Exception as e:
        print(f"❌ Error evaluating checkpoint: {e}")
        return None