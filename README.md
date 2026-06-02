# From Manuals to Reasoning Traces


Project to convert natural language procedural documents (repair manuals, medcial protocols, military SOP) into structured workflow graphs, then automatically generate training data 
to teach a small planning agent to follow those procedures step by step. No manual annotation needed in the pipeline.

The motivation for this is very simple: organizations have procedures written in Word docs and PDFs their employees need to follow. This pipeline reads them, 
extracts the workflow structure (actions, gateways, branches), generates positive and negative training examples (near-miss cases to teach the agent what can go wrong),
and fine-tunes a small planner on top of Llama 3.1 8B that can actually execute those procedures. Since a black-box model has no idea about orgnizations own proprietary data, the pipeline 
needs to be grounded in their own KB, be factual and never hallucinate. Relying on a black-box model would make training of specialized agents impossible, costs high and would make your
organization fully-dependedt on the Big AI players - when they release a new model, your system would already be outdated. 



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

| Method | What it gets | 
|--------|-------------|
| M1: Bare Llama | Just the procedure text 
| M2: Llama + Actions | + extracted action list 
| M3: Ensemble | + PRM blend (α=0.9) + extracted action list 
| M4: Agentic Ensemble | + graph tool when uncertain + extracted action list|

We show that extraction quality is the bottleneck, not the agent via experiemnts.

Zero hallucinations from Method 2 onward so we finally achieve a desired outcome by combining both planning
agents and extraction agent into one unified method(vs 35.4% for bare Llama).

## Key Findings

- **Extraction quality is the bottleneck.** Agent on gold graphs: 98%. Same agent on extracted graphs: 55%. Fix the extraction, fix the agent.
- **Less training data can be better.** The deduplicated PRM (43% less data) outperforms the full one on the ensemble config (+8.2% completion on gold).
- **You don't need a massive model.** An 8B model with a LoRA adapter, combined with the right pipeline components, gets the job done.
- **Each component matters.** Action list eliminates hallucinations. PRM improves planning. Graph tool helps at gateways. Remove any one and performance drops.

## Dataset
Built on [PAGED](https://github.com/HLR/PAGED) — 489 test procedures covering business processes, 
medical workflows, and operational procedures. Gold workflow graphs generated from the dataset's 
SequenceFlow representation.

## My Trained Models

- PRM (full data): [huggingface.co/alexbalan08/PRM](https://huggingface.co/alexbalan08/PRM)
- PRM (deduplicated): [huggingface.co/alexbalan08/Procedural-small](https://huggingface.co/alexbalan08/Procedural-small)

