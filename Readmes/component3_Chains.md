-Used to build pipelines
-Automatically makes output of prev stage to next stages input. No manual code needed.

Chains Types

$Parallel Chain : Parallel Chains in LangChain allow multiple independent LLM chains to execute simultaneously and return results efficiently. 
                  This is useful when dealing with:
                  ✅ Multiple tasks at once (e.g., translation + summarization)
                  ✅ Fetching data from different sources
                  ✅ Parallel processing of independent inputs
$Conditional Chain : Conditional Chains in LangChain allow for dynamic decision-making in workflows. Based on specific conditions, the execution path changes, meaning different logic or chains can be triggered.
                    🔹 Why Use Conditional Chains?
                    ✅ When different inputs require different processing paths
                    ✅ To dynamically select which chain should execute
                    ✅ To create if-else logic inside LangChain workflows
