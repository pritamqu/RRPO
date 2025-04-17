




if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, default=None, 
                        choices=['videochat2_mistral_7b', 'llavavideo_qwen_7b', 'longvu_qwen_7b'])
    parser.add_argument('--model-path', default='', required=True)
    parser.add_argument('--model-path2', help='', default=None, type=str, required=False)
    parser.add_argument("--video_path", type=str, default='sample_video.mp4')
    parser.add_argument('--question', default='Describe this video.')     
    parser.add_argument("--model_max_length", type=int, required=False, default=1024)
    args = parser.parse_args()

    # model
    if args.base_model_name == "videochat2_mistral_7b":
        from videochat2.load_videochat2 import load_model, \
            get_model_output, load_lora_weights
        model, resolution, num_frame = load_model(args.model_path)
        if args.model_path2:
            model = load_lora_weights(args.model_path2, model)

    elif args.base_model_name == "llavavideo_qwen_7b":
        from llava.load_llavavideo import load_model, \
            get_model_output, load_lora_weights
        tokenizer, model, image_processor = load_model(args.model_path)
        if args.model_path2:
            model = load_lora_weights(args.model_path2, model)

        num_frame=64

    elif args.base_model_name == "longvu_qwen_7b":
        model_name="cambrian_qwen"
        version='qwen'
        from longvu.builder import load_pretrained_model as load_model
        from longvu.load_longvu import get_model_output, load_lora_weights
        tokenizer, model, image_processor, context_len = load_model(
                                        args.model_path,  
                                        None,
                                        model_name,
                                        device_map=None,
                                    )
        if args.model_path2:
            model = load_lora_weights(args.model_path2, model)
        
        model.config.use_cache = True
        model.cuda()

    else:
        raise NotImplementedError(args.base_model_name)
    
    video_path=args.video_path
    question=args.question

    # run inference
    if args.base_model_name == "videochat2_mistral_7b":
        system_msg="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, answer the question.\n"
        outputs=get_model_output(model, 
                                    video_path, 
                                    num_frame, 
                                    resolution, 
                                    question, 
                                    args.model_max_length, 
                                    device='cuda', 
                                    system=system_msg,
                                    system_q=False,
                                    system_llm=True,
                                    question_prompt='',
                                    answer_prompt=None,)

    elif args.base_model_name == "llavavideo_qwen_7b":
        outputs=get_model_output(model, 
                    tokenizer,
                    image_processor,
                    video_path, 
                    num_frame=num_frame, 
                    question=question, 
                    max_new_tokens=args.model_max_length, 
                    temperature=0.0)

    elif args.base_model_name == "longvu_qwen_7b":
        outputs=get_model_output(model, 
                    tokenizer,
                    image_processor,
                    video_path, 
                    num_frame=-1,
                    question=question, 
                    max_new_tokens=args.model_max_length, 
                    temperature=0.0,
                    version=version)
        
    else:
        raise NotImplementedError()
    

    print(f"Question: {question}\nResponse: {outputs}")
