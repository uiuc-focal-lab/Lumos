import os
import argparse
from statsmodels.stats.proportion import proportion_confint
from qwen_inference import Qwen2VLInference
from llava_inference import LlavaInference

def get_model(model_name):
    if "qwen" in model_name.lower():
        return Qwen2VLInference()
    elif "llava" in model_name.lower():
        return LlavaInference()
    else:
        raise ValueError(f"Unknown model: {model_name}")

def check_answer(model_output, correct_answer):
    return correct_answer.lower() in model_output.lower()

def certify(question, answer, image_dir, model_name):
    model = get_model(model_name)
    
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()
    
    total = len(images)
    if total == 0:
        print("No images found in directory.")
        return None

    correct = 0
    results = []

    print(f"Running certification on {total} images using {model_name}...")
    print(f"Question: {question}")
    print(f"Expected Answer: {answer}")

    for i, img_file in enumerate(images):
        img_path = os.path.join(image_dir, img_file)
        
        try:
            if "qwen" in model_name.lower():
                output = model.infer_w_qwen(img_path, question)
                # Qwen returns a list of strings
                if isinstance(output, list):
                    output = output[0]
            elif "llava" in model_name.lower():
                output = model.infer(img_path, question)
            else:
                output = ""
            
            is_correct = check_answer(output, answer)
            if is_correct:
                correct += 1
                
            results.append({
                "image": img_file,
                "output": output,
                "correct": is_correct
            })
            
            print(f"[{i+1}/{total}] Image: {img_file} | Correct: {is_correct} | Output: {output.strip()}")
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

    # Calculate confidence intervals
    lower, upper = proportion_confint(correct, total, alpha=0.05, method='beta')
    
    return {
        "correct": correct,
        "total": total,
        "lower_bound": lower,
        "upper_bound": upper,
        "results": results
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Certify VQA model performance.")
    parser.add_argument("--question", type=str, required=True, help="The question to ask.")
    parser.add_argument("--answer", type=str, required=True, help="The expected answer.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (qwen or llava).")
    
    args = parser.parse_args()
    
    result = certify(args.question, args.answer, args.image_dir, args.model_name)
    
    if result:
        print("\n" + "="*50)
        print("CERTIFICATION RESULTS")
        print("="*50)
        print(f"Model: {args.model_name}")
        print(f"Total Images: {result['total']}")
        print(f"Correct: {result['correct']}")
        print(f"Accuracy: {result['correct']/result['total']:.2%}")
        print(f"95% Confidence Interval: ({result['lower_bound']:.4f}, {result['upper_bound']:.4f})")
        print("="*50)
