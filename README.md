# From Manuals to Reasoning Traces


Project to convert natural language procedural documents (repair manuals, medcial protocols, military SOP) into structured workflow graphs, then automatically generate training data 
to teach a small planning agent to follow those procedures step by step. No manual annotation needed in the pipeline.

The motivation for this is very simple: organizations have procedures written in Word docs and PDFs their employees need to follow. This pipeline reads them, 
extracts the workflow structure (actions, gateways, branches), generates positive and negative training examples (near-miss cases to teach the agent what can go wrong),
and fine-tunes a small planner on top of Llama 3.1 8B that can actually execute those procedures. Since a black-box model has no idea about orgnizations own proprietary data, the pipeline 
needs to be grounded in their own KB, be factual and never hallucinate. Relying on a black-box model would make training of specialized agents impossible, costs high and would make your
organization fully-dependedt on the Big AI players - when they release a new model, your system would already be outdated. 


### Data Flywheel 
The full process is iterative since every new procedure extracted at deployment produces a new graph and more training data, so the planner improves over time without manual labeling. This is a data flywheel that keeps up with an organization's new proprietary procedures. extraction phase is made specifically so there is no fine-tunning so a large set of annotated procedures would not actually be required. Also the costs of extraction are very low which was another goal. 



### Extraction Pipeline
## what we miss: 
The first step of actually converting those PDFs into JSON format so we could start our extraaction pipeline - ByteIT.ai from europe 
can provide such service with good accuracy
## what we have 
- Takes natural language procedure text and produces a structured JSON or YAML workflow graph (we show why JSON is superior to YAML. Literature takes care
- of other representations) 
- Uses GPT-5.4 Mini with a self-refinement loop (structural checker + LLM semantic checker and reflexion memeory to learn from past feedback)
- RAG retrieval from a pool of gold-labeled examples (we test different configurations)
- Reflexion memory that accumulates past extraction errors across procedures
- Achieves 0.878 Action F1, 0.644 Edge F1, 0.958 Gateway Type Accuracy on 464 test procedures - it's surprisningly good for linear procedures


### Training Data Construction
- Converts extracted graphs into execution states (all valid paths through the procedure that can be taken)
- Generates 5 types of near-miss negatives to teach the model cases which might seem plausible, but they can actually produce wrong executions:
  each targets a specific agent failure mode we actually observed during manual validation
- Produces 14,651 step-level SFT records (8,628 positive, 6,023 negative)
- Deduplication pass removes 43% near-duplicate cases and fully balances the positive-negative examples 
- The cleaned set actually performs better than the full set on some confiugrations, and close on the other.This is a big win, since less data for training
- already means reduced costs for any organization

### Planning PRM Training
- LoRA adapter on Llama 3.1 8B Instruct (4-bit quantized so inference could be done locally on a simple NVIDIA GPU. 8GB Vram might even be enough)
- Trained on A100 via Google Colab 
- Two variants: full data (14,651) and deduplicated (8,400)
- Both adapters available on HuggingFace (links below)

### Agent Configurations
Four methods, each adding one layer so we could properly isolate the effects of each added component:

| Method | What it gets | All methods include the previous components by default |
|--------|-------------|
| M1: Bare Llama | Just the procedure text 
| M2: Llama + Actions | + extracted action list 
| M3: Ensemble | + PRM blend (α=0.9) 
| M4: Agentic Ensemble | + graph tool when uncertain |

We show that extraction quality is the bottleneck, not the agent via experiemnts.

Zero hallucinations from Method 2 onward so we finally achieve a desired outcome by combining both planning
agents and extraction agent into one unified method(vs 35.4% for bare Llama).

## Some interesting findings

- **Extraction quality is the bottleneck.** we need better quality extraction, agents for procedures following perform really well if the correct information is passed to them via tool-use.
- **Less training data can be better.** The deduplicated PRM (43% less data) outperforms the full one on the ensemble config
- **You don't need a massive and expensive  model.** A mix of 8B models, with a specialized LoRA adapter on top, combined with the right pipeline components, could even outperform a SOTA black-box model - maybe we show this in the future :).
- **Each component matters.** Action list eliminates hallucinations. PRM improves planning. Graph tool helps probably at gateways

## Dataset
Built on PAGED procedural-documents dataset (the largest annoated one out there)


PAGED: A Benchmark for Procedural Graphs Extraction from Documents, 

Weihong Du and Wenrui Liao and Hongru Liang and Wenqiang Lei

2024

## My Trained Models

- PRM (full data): [huggingface.co/alexbalan08/PRM](https://huggingface.co/alexbalan08/PRM)
- PRM (deduplicated): [huggingface.co/alexbalan08/Procedural-small](https://huggingface.co/alexbalan08/Procedural-small)

## How to run:

- First install all the requirments.
- Ideally use Linux for fast inference + you will need a GPU with around 16GB VRAM (maybe 8 is enough)
- You will need an OpenAI API key for the extractor base model (very cheap)
- First run final extracxtion, then build TR-data including negative examples
- Then use the trained model at inference on the test set of procedures or new one and see results 

## Master Thesis
Maastricht University, Department of Advanced Computing Sciences


