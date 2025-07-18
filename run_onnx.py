
from catt_models_onnx import CATTEncoderDecoder, CATTEncoderOnly

ed_model_folder = "onnx_models/ed_model"
eo_model_folder = "onnx_models/eo_model"

ed_model = CATTEncoderDecoder(encoder_path=f'{ed_model_folder}/encoder.onnx',
                              decoder_path=f'{ed_model_folder}/decoder.onnx')
eo_model = CATTEncoderOnly(encoder_path=f'{eo_model_folder}/encoder.onnx',
                           decoder_path=f'{eo_model_folder}/decoder.onnx')


x = ['وقالت مجلة نيوزويك الأمريكية التحديث الجديد ل إنستجرام يمكن أن يساهم في إيقاف وكشف الحسابات المزورة بسهولة شديدة']

batch_size = 16
verbose = False
for onnx_model in [ed_model, eo_model]:
    print('Model:', type(onnx_model))
    # Use exactly like your original model
    result = onnx_model.do_tashkeel(x[0])
    results = onnx_model.do_tashkeel_batch(x, batch_size, verbose)

    print(result)
    print(results)
