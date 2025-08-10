**Temperature** :Parameter that controls randomness of a language model's output- varies between 0-2.
Its a creativity parameter> how much creative response you need in your model.
*Lower values(0.0-0.3)- More deterministic & predictable*
*Higher values(0.7-1.5)- More creative & unpredictable*

$Use-Case:
Factual answers(math, code, facts):0.0-0.3
Balanced response(general QA, explanations):0.5-0.7
Creative writing etc : 0.9-1.2
Max randomness : 1.5+


max_completion_tokens: how much max token needed in output