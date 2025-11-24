import os
import argparse
from gliner import GLiNER

def main(args):
    gliner_model = GLiNER.from_pretrained(args.model_path)

    gliner_model.export_to_onnx(save_dir = args.save_path, 
                                onnx_filename=args.file_name, 
                                quantized_filename=args.quantized_file_name,
                                quantize=args.quantize,
                                opset=args.opset)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default= "logs/model_12000")
    parser.add_argument('--save_path', type=str, default = 'model/')
    parser.add_argument('--file_name', type=str, default = 'models.onnx')
    parser.add_argument('--quantized_file_name', type=str, default = 'models_quantized.onnx')
    parser.add_argument('--opset', type=int, default = 19)
    parser.add_argument('--quantize', type=bool, default = True)
    args = parser.parse_args()
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    main(args)

    print("Done!")