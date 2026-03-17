from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang='ch',
    det_model_dir='Module/ch_ppocr_mobile_v2.0_det_infer',
    cls_model_dir='Module/ch_ppocr_mobile_v2.0_cls_infer',
    rec_model_dir='Module/ch_ppocr_mobile_v2.0_rec_infer'
)

print("init ok")