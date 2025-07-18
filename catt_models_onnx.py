import os
import math
import numpy as np
from tqdm import tqdm
from pathlib import Path
import onnxruntime as ort
from abc import ABC, abstractmethod
from utils import remove_non_arabic
from tashkeel_tokenizer_onnx import TashkeelTokenizer


class BaseONNXTashkeel(ABC):
    """
    Base class for ONNX-based Arabic text diacritization models

    This abstract base class provides common functionality for Arabic text
    diacritization using pre-trained ONNX models.
    """

    def __init__(self, encoder_path=None, decoder_path=None, tokenizer=None,
                 auto_preprocess=True, model_type="ed_model"):
        """
        Initialize ONNX Tashkeel model

        Args:
            encoder_path (str, optional): Path to encoder ONNX file
            decoder_path (str, optional): Path to decoder ONNX file
            tokenizer (TashkeelTokenizer, optional): Tokenizer instance
            auto_preprocess (bool): Automatically preprocess text (remove non-Arabic)
            model_type (str): Type of model ("ed_model" or "eo_model")
        """
        self.tokenizer = tokenizer or TashkeelTokenizer()
        self.auto_preprocess = auto_preprocess
        self.src_pad_idx = self.tokenizer.letters_map['<PAD>']

        # Set model-specific paths
        encoder_path, decoder_path = self._get_model_paths(
            encoder_path, decoder_path, model_type
        )

        # Load ONNX sessions
        self._load_onnx_sessions(encoder_path, decoder_path)

    def _get_model_paths(self, encoder_path, decoder_path, model_type):
        """Get default model paths if not provided"""
        if encoder_path is None or decoder_path is None:
            package_dir = Path(__file__).parent
            model_dir = package_dir / "onnx_models" / model_type
            encoder_path = encoder_path or str(model_dir / "encoder.onnx")
            decoder_path = decoder_path or str(model_dir / "decoder.onnx")
        return encoder_path, decoder_path

    def _load_onnx_sessions(self, encoder_path, decoder_path):
        """Load ONNX runtime sessions"""
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder ONNX file not found: {encoder_path}")
        if not os.path.exists(decoder_path):
            raise FileNotFoundError(f"Decoder ONNX file not found: {decoder_path}")

        self.encoder_session = ort.InferenceSession(encoder_path)
        self.decoder_session = ort.InferenceSession(decoder_path)
        print("✅ Loaded model successfully!")

    def _preprocess_text(self, text):
        """Preprocess text if auto_preprocess is enabled"""
        return remove_non_arabic(text) if self.auto_preprocess else text

    def _make_pad_mask(self, q, k, q_pad_idx, k_pad_idx):
        """Create padding mask (numpy version)"""
        len_q, len_k = q.shape[1], k.shape[1]
        k_mask = (k != k_pad_idx).reshape(-1, 1, 1, len_k)
        k_mask = np.repeat(k_mask, len_q, axis=2)
        q_mask = (q != q_pad_idx).reshape(-1, 1, len_q, 1)
        q_mask = np.repeat(q_mask, len_k, axis=3)
        return k_mask & q_mask

    def _make_no_peak_mask(self, q, k):
        """Create no-peak mask (numpy version)"""
        len_q, len_k = q.shape[1], k.shape[1]
        return np.tril(np.ones((len_q, len_k))).astype(bool)

    def _run_encoder(self, src_np):
        """Run encoder using ONNX"""
        src_mask_np = self._make_pad_mask(
            src_np, src_np, self.src_pad_idx, self.src_pad_idx
        ).astype(bool)

        encoder_inputs = {'src': src_np, 'src_mask': src_mask_np}
        return self.encoder_session.run(None, encoder_inputs)[0]

    def _pad_seq(self, sequences, batch_first=True, padding_value=0.0, prepadding=True):
        """Pad sequences to same length"""
        lens = [seq.shape[0] for seq in sequences]
        max_len = max(lens)

        # Create padded array
        if len(sequences[0].shape) == 1:
            padded_sequences = np.full((len(sequences), max_len), padding_value, dtype=sequences[0].dtype)
        else:
            shape = (len(sequences), max_len) + sequences[0].shape[1:]
            padded_sequences = np.full(shape, padding_value, dtype=sequences[0].dtype)

        # Fill with actual sequence data
        for i, seq in enumerate(sequences):
            padded_sequences[i, :lens[i]] = seq

        if prepadding:
            for i in range(len(lens)):
                padded_sequences[i] = np.roll(padded_sequences[i], max_len - lens[i], axis=0)

        if not batch_first:
            padded_sequences = padded_sequences.transpose(1, 0, *range(2, padded_sequences.ndim))

        return padded_sequences

    def _get_batches(self, X, batch_size=16):
        """Generate batches from list"""
        num_batches = math.ceil(len(X) / batch_size)
        for i in range(num_batches):
            yield X[i*batch_size : (i+1)*batch_size]

    def _prepare_batch_input(self, texts_mini):
        """Prepare input batch for processing"""
        input_ids_list = []
        for text in texts_mini:
            input_ids, _ = self.tokenizer.encode(text, test_match=False)
            input_ids_list.append(input_ids)

        return self._pad_seq(
            input_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.letters_map['<PAD>'],
            prepadding=False
        )

    def _apply_space_mask(self, predictions, input_ids):
        """Apply space masking to predictions"""
        space_mask = (input_ids == self.tokenizer.letters_map[' '])
        predictions[space_mask] = self.tokenizer.tashkeel_map[self.tokenizer.no_tashkeel_tag]
        return predictions

    @abstractmethod
    def _process_batch(self, batch_input_ids):
        """Process a batch of input IDs - to be implemented by subclasses"""
        pass

    def do_tashkeel_batch(self, texts, batch_size=16, verbose=True):
        """
        Add tashkeel to multiple Arabic texts

        Args:
            texts (list): List of input Arabic texts
            batch_size (int): Batch size for processing
            verbose (bool): Show progress bar

        Returns:
            list: List of texts with tashkeel
        """
        processed_texts = [self._preprocess_text(text) for text in texts]
        text_with_tashkeel = []
        data_iter = self._get_batches(processed_texts, batch_size)

        if verbose:
            num_batches = math.ceil(len(processed_texts) / batch_size)
            data_iter = tqdm(data_iter, total=num_batches)

        for texts_mini in data_iter:
            batch_input_ids = self._prepare_batch_input(texts_mini)
            predictions = self._process_batch(batch_input_ids)
            text_with_tashkeel_mini = self.tokenizer.decode(batch_input_ids, predictions)
            text_with_tashkeel.extend(text_with_tashkeel_mini)

        return text_with_tashkeel

    def do_tashkeel(self, text, verbose=True):
        """
        Add tashkeel to Arabic text

        Args:
            text (str): Input Arabic text

        Returns:
            str: Text with tashkeel
        """
        return self.do_tashkeel_batch([text], verbose=False)[0]


class CATTEncoderDecoder(BaseONNXTashkeel):
    """
    ONNX-based Arabic text diacritization model using encoder-decoder architecture
    """

    def __init__(self, encoder_path=None, decoder_path=None, tokenizer=None,
                 auto_preprocess=True):
        """Initialize encoder-decoder model"""
        super().__init__(encoder_path, decoder_path, tokenizer, auto_preprocess, "ed_model")
        self.trg_pad_idx = self.tokenizer.tashkeel_map['<PAD>']

    def _run_decoder(self, trg_np, enc_src_np, src_np):
        """Run decoder using ONNX"""
        src_trg_mask_np = self._make_pad_mask(
            trg_np, src_np, self.trg_pad_idx, self.src_pad_idx
        ).astype(bool)

        trg_mask_np = self._make_pad_mask(
            trg_np, trg_np, self.trg_pad_idx, self.trg_pad_idx
        ).astype(bool)

        no_peak_mask_np = self._make_no_peak_mask(trg_np, trg_np)
        trg_mask_np = trg_mask_np & no_peak_mask_np

        decoder_inputs = {
            'trg': trg_np,
            'enc_src': enc_src_np,
            'trg_mask': trg_mask_np,
            'src_trg_mask': src_trg_mask_np
        }

        return self.decoder_session.run(None, decoder_inputs)[0]

    def _process_batch(self, batch_input_ids):
        """Process batch using encoder-decoder with autoregressive decoding"""
        src = batch_input_ids
        enc_src = self._run_encoder(src)

        # Initialize target with BOS token
        target_ids = np.array([[self.tokenizer.tashkeel_map['<BOS>']]] * len(batch_input_ids), dtype=np.int64)

        # Autoregressive decoding loop
        for i in range(src.shape[1] - 1):
            preds = self._run_decoder(target_ids, enc_src, src)
            next_tokens = np.argmax(preds[:, -1, :], axis=1)
            target_ids = np.concatenate([target_ids, np.expand_dims(next_tokens, 1)], axis=1)

            # Apply space handling logic
            target_ids = self._apply_space_mask(target_ids, src[:, :target_ids.shape[1]])

        return target_ids


class CATTEncoderOnly(BaseONNXTashkeel):
    """
    ONNX-based Arabic text diacritization model using encoder-only architecture
    """

    def __init__(self, encoder_path=None, decoder_path=None, tokenizer=None,
                 auto_preprocess=True):
        """Initialize encoder-only model"""
        super().__init__(encoder_path, decoder_path, tokenizer, auto_preprocess, "eo_model")

    def _run_decoder(self, enc_src_np):
        """Run decoder using ONNX"""
        decoder_inputs = {'enc_src': enc_src_np}
        return self.decoder_session.run(None, decoder_inputs)[0]

    def _process_batch(self, batch_input_ids):
        """Process batch using encoder-only architecture"""
        # Remove BOS and EOS tokens
        batch_input_ids = batch_input_ids[:, 1:-1]

        enc_src = self._run_encoder(batch_input_ids)
        preds = self._run_decoder(enc_src)
        predictions = np.argmax(preds, axis=-1)

        # Apply space masking
        predictions = self._apply_space_mask(predictions, batch_input_ids)

        return predictions
