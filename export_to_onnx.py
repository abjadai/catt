
import os
import torch
import torch.nn as nn
from pathlib import Path


def export_tashkeel_model_to_onnx(
    pl_model,
    tokenizer,
    output_dir="onnx_models",
    encoder_filename="encoder.onnx",
    decoder_filename="decoder.onnx"
):
    """
    Export TashkeelModel (PyTorch Lightning) to separate ONNX encoder and decoder models

    Args:
        pl_model: TashkeelModel instance (PyTorch Lightning model)
        tokenizer: Tokenizer instance used with the model
        output_dir: Directory to save ONNX models
        encoder_filename: Name for encoder ONNX file
        decoder_filename: Name for decoder ONNX file

    Returns:
        dict: Paths to exported ONNX files
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir)

    encoder_path = output_path / encoder_filename
    decoder_path = output_path / decoder_filename

    # Set model to evaluation mode
    pl_model.eval()

    # Export encoder
    print("Exporting encoder to ONNX...")
    _export_encoder(pl_model, tokenizer, encoder_path)

    # Export decoder
    print("Exporting decoder to ONNX...")
    _export_decoder(pl_model, tokenizer, decoder_path)

    print(f"✅ Encoder exported to: {encoder_path}")
    print(f"✅ Decoder exported to: {decoder_path}")

    return {
        "encoder_path": str(encoder_path),
        "decoder_path": str(decoder_path),
        "tokenizer": tokenizer
    }


def _export_encoder(pl_model, tokenizer, encoder_path, max_len=1024):
    """Export encoder to ONNX"""
    # Get model parameters
    batch_size = 1

    # Create dummy inputs for encoder
    dummy_src = torch.randint(0, len(tokenizer.letters_map), (batch_size, max_len))
    dummy_src_mask = torch.ones(batch_size, 1, max_len, max_len, dtype=torch.bool)

    # Create encoder wrapper
    class EncoderWrapper(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, src, src_mask):
            return self.encoder(src, src_mask)

    encoder_wrapper = EncoderWrapper(pl_model.transformer.encoder)
    encoder_wrapper.eval()

    # Export encoder
    with torch.no_grad():
        torch.onnx.export(
            encoder_wrapper,
            (dummy_src, dummy_src_mask),
            encoder_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['src', 'src_mask'],
            output_names=['encoder_output'],
            dynamic_axes={
                'src': {0: 'batch_size', 1: 'src_seq_len'},
                'src_mask': {0: 'batch_size', 2: 'src_seq_len', 3: 'src_seq_len'},
                'encoder_output': {0: 'batch_size', 1: 'src_seq_len'}
            }
        )


def _export_decoder(pl_model, tokenizer, decoder_path, max_len=1024, d_model=512):
    """Export decoder to ONNX"""
    # Get model parameters
    batch_size = 1

    # Create dummy inputs for decoder
    dummy_trg = torch.randint(0, len(tokenizer.tashkeel_map), (batch_size, max_len))
    dummy_enc_src = torch.randn(batch_size, max_len, d_model)
    dummy_trg_mask = torch.ones(batch_size, 1, max_len, max_len, dtype=torch.bool)
    dummy_src_trg_mask = torch.ones(batch_size, 1, max_len, max_len, dtype=torch.bool)

    is_eo = isinstance(pl_model.transformer.decoder, nn.Linear)

    if is_eo:
        print('CATT Encoder Only model detected!')
        # Create decoder wrapper
        class DecoderWrapper(nn.Module):
            def __init__(self, decoder):
                super().__init__()
                self.decoder = decoder

            def forward(self, enc_src):
                return self.decoder(enc_src)

    else:
        print('CATT EncoderDecoder model detected!')
        # Create decoder wrapper
        class DecoderWrapper(nn.Module):
            def __init__(self, decoder):
                super().__init__()
                self.decoder = decoder

            def forward(self, trg, enc_src, trg_mask, src_trg_mask):
                return self.decoder(trg, enc_src, trg_mask, src_trg_mask)

    decoder_wrapper = DecoderWrapper(pl_model.transformer.decoder)
    decoder_wrapper.eval()

    if is_eo:
        decoder_input = (dummy_enc_src,)
        input_names = ['enc_src']
        output_names = ['decoder_output']
        dynamic_axes = {
            'enc_src': {0: 'batch_size', 1: 'src_seq_len'},
            'decoder_output': {0: 'batch_size', 1: 'src_seq_len'}
        }
    else:
        decoder_input = (dummy_trg, dummy_enc_src, dummy_trg_mask, dummy_src_trg_mask)
        input_names = ['trg', 'enc_src', 'trg_mask', 'src_trg_mask']
        output_names = ['decoder_output']
        dynamic_axes = {
            'trg': {0: 'batch_size', 1: 'trg_seq_len'},
            'enc_src': {0: 'batch_size', 1: 'src_seq_len'},
            'trg_mask': {0: 'batch_size', 2: 'trg_seq_len', 3: 'trg_seq_len'},
            'src_trg_mask': {0: 'batch_size', 2: 'trg_seq_len', 3: 'src_seq_len'},
            'decoder_output': {0: 'batch_size', 1: 'trg_seq_len'}
        }

    # Export decoder
    with torch.no_grad():
        torch.onnx.export(
            decoder_wrapper,
            decoder_input,
            decoder_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )


def validate_onnx_models(encoder_path, decoder_path):
    """
    Validate that ONNX models were exported correctly

    Args:
        encoder_path: Path to encoder ONNX file
        decoder_path: Path to decoder ONNX file

    Returns:
        bool: True if validation passes
    """
    try:
        import onnx
        import onnxruntime as ort

        # Load and check ONNX models
        encoder_model = onnx.load(encoder_path)
        decoder_model = onnx.load(decoder_path)

        # Check model validity
        onnx.checker.check_model(encoder_model)
        onnx.checker.check_model(decoder_model)

        # Test ONNX Runtime sessions
        encoder_session = ort.InferenceSession(encoder_path)
        decoder_session = ort.InferenceSession(decoder_path)

        print("✅ ONNX models validation passed")
        return True

    except Exception as e:
        print(f"❌ ONNX validation failed: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    import torch
    from eo_pl import TashkeelModel as TashkeelModelEO
    from ed_pl import TashkeelModel as TashkeelModelED
    from tashkeel_tokenizer import TashkeelTokenizer
    from utils import remove_non_arabic

    print('Exporting CATT EncoderOnly model to ONNX...')

    tokenizer = TashkeelTokenizer()
    ckpt_path = 'models/best_eo_mlm_ns_epoch_193.pt'

    print('ckpt_path is:', ckpt_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    max_seq_len = 1024
    print('Creating Model...')
    model = TashkeelModelEO(tokenizer, max_seq_len=max_seq_len, n_layers=6, learnable_pos_emb=False)

    print(model.load_state_dict(torch.load(ckpt_path, map_location=device)))
    model.eval()

    # Export to ONNX
    export_info = export_tashkeel_model_to_onnx(
        pl_model=model,
        tokenizer=tokenizer,
        output_dir="onnx_models/eo_model"
    )

    # Validate exported models
    validate_onnx_models(
        export_info["encoder_path"],
        export_info["decoder_path"]
    )

    print("Export completed successfully!")


    print('Exporting CATT EncoderDecoder model to ONNX...')

    ckpt_path = 'models/best_ed_mlm_ns_epoch_178.pt'

    print('ckpt_path is:', ckpt_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    max_seq_len = 1024
    print('Creating Model...')
    model = TashkeelModelED(tokenizer, max_seq_len=max_seq_len, n_layers=3, learnable_pos_emb=False)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Export to ONNX
    export_info = export_tashkeel_model_to_onnx(
        pl_model=model,
        tokenizer=tokenizer,
        output_dir="onnx_models/ed_model"
    )

    # Validate exported models
    validate_onnx_models(
        export_info["encoder_path"],
        export_info["decoder_path"]
    )

    print("Export completed successfully!")
