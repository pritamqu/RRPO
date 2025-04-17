


### How to use your own data?

The dataloader for LLaVA-Video and LongVU are setup to load `json` files with samples structured as:

```
{
    "question": "<image>\nThis is a sample question.,
    "answer": "This is a correct or preferred answer.",
    "pred": "This is an incorrect or non-preferred answer.",
    "answer_masked": "This is a <MASK> correct </MASK> or <MASK> preferred </MASK> answer.",
    "pred_masked": "This is an <MASK> incorrect </MASK> or <MASK> non-preferred </MASK> answer.",
    "src": "data-source",
    "sample_id": 0,
    "video": "video.mp4"
},

```

You can structure your samples to train models with RRPO loss. RRPO loss is designed to penalize phrases inside `<MASK>` and `</MASK>` tags. Assume you have a pair of preferred and non-preferred responses, you can employ a powerful LLM to do this. Alternatively, if the responses are fairly aligned, you can also use regex based approach. We use LLM for open-ended long reponse, and use regex for mcq, binary qa, and short answers. Here is a sample prompt that we used in our case. You can modify this as per your requirement.

**Sample prompt**

*first turn*
```
Identify the key differences between these two sentences.
To identify differences focus on the key aspects in the sentences, 
such as objects, actions, and their attributes, among others.
If there are no key difference between these two sentences, 
please respond with "None" and nothing else.

Sentence 1: {sentence_from_ground_truth}
Sentence 2: {sentence_from_generated_response}
```

*second turn*
```
Please rewrite the "Sentence 1" by incorporating the differences you 
mentioned earlier. Your final response should contain only the revised 
sentence and nothing else.

Sentence 1: {sentence_from_ground_truth}
```