Table of Contents
Session 5: Data Mixtures and Curriculum
1. What this session is

In the last two sessions, we understood the slow and unglamorous work required to turn the open web into something a model can actually learn from. By the end of Session 4, we had clean, deduplicated and provenance-stamped shards, with each shard tagged by where it came from and what kind of data it theoretically contains.

This session is where those shards stop being a pile of clean data and become a training plan. A corpus does not decide what the model becomes. That decision comes from the data mixture: how much of each category the model sees, how often it sees it and at what stage of training it sees it. The same clean corpus and the same compute budget can produce completely different models depending on this mixture. This is where the capabilities and personality of the model are actually decided.

The target is clear. We want a model that is excellent at coding and agentic work, similar to a Codex-style assistant. It should be able to take a long task, plan the work, call tools across multiple steps, read the results, recover when a call fails and continue operating while holding the growing history of the task in context. We also want a strong reasoning model whose depth of reasoning can be controlled. Simple problems should require little reasoning, while difficult problems should allow the model to think for longer and in greater depth.

Finally, the model must understand and generate Indic languages natively. This is the primary differentiator and the reason this project needs to exist. The work of this session is to compose a data mixture that deliberately produces these capabilities, and then arrange that mixture into a curriculum the model can absorb reliably.

2. Why the mixture is the model

The easiest way to understand that the mixture is a real design decision is to change it and observe how the model changes. These proportions are not bookkeeping values that can be set once and ignored.

If more of the fixed token budget is allocated to code and less to general web text, the model becomes better at programming but loses some breadth of world knowledge. If a fixed share is reserved for Indic languages, that capability remains protected. Without this protection, the English-heavy distribution of the web will gradually push Indic data out of the mixture.

The total token budget is fixed. Every additional share given to one capability is taken from another. Designing the mixture is therefore the process of deciding which capabilities matter enough to justify that trade-off.

A V5 mixture composer where every capability receives a share of the fixed token budget. Moving one slider automatically renormalizes the others, so the total always remains 100%. The default preset shows the main pretraining mixture, where general web remains the largest lane because it is the most abundant source of data. A separate annealing preset shows the concentrated final phase, where general web is reduced and scarce capabilities are deliberately upsampled. As the mixture changes, the composer highlights the benchmarks influenced by each lane and compares the required tokens against the real supply available. This immediately exposes the agentic lane: very little high-quality agentic data exists naturally, so much of it must be synthesized. Protected floors prevent the Indic and agentic shares from falling below their minimum allocation. Loading the naive web-heavy preset shows both capabilities collapsing as general web expands, capturing the central lesson of this session in one interaction.

The composer primarily represents the main pretraining run. General web receives the largest share because the open internet contains far more of it than code, mathematics, Indic or agentic trajectories. The capabilities we care about may be strategically important, but the data required to teach them is scarce. We therefore preserve the highest-quality scarce data for a short and concentrated annealing phase near the end of training. In this phase, general web falls sharply while code, reasoning, Indic and agentic data receive much larger shares. This final mixture is separate from the main pretraining mixture and must be treated as its own preset. The supply check keeps the design honest by showing which allocations can be met using unique tokens, which require repetition and which can only be reached through synthetic generation. The agentic lane must largely be built rather than collected.

This is not merely a theoretical argument. V4 already followed this principle. Across its growth stages, the mixture was deliberately rebalanced. General web fell from roughly 70% toward 18%. Code increased from approximately 13% to 35%, while science and mathematics increased from around 7% to 39%. A protected channel remained fixed at 8% throughout the run. These were explicit training decisions. The capabilities of the final model were the capabilities those allocations purchased.

3. Composing backward from the benchmarks

If the mixture is a set of trade-offs between capabilities, then those trade-offs should be made by working backward from the capabilities we actually intend to demonstrate. A capability that is never measured is a capability nobody can verify was built.

In practice, this means starting from the benchmarks the model will be judged on and mapping each benchmark to the data required to improve it. The mixture then becomes a deliberate answer to what we are trying to win, rather than a collection of round numbers that merely felt reasonable. This only works when we understand what each benchmark is actually testing. Much of the confusion begins here, because benchmark names are repeated constantly even when people have never examined a single example of what the model is being asked to do.

A benchmark explainer covering every benchmark used in this session, grouped into agentic and tool use, coding, reasoning and mathematics, and Indic. Selecting a benchmark shows what it measures, how it is scored and a real sample task, so the model’s required output is visible rather than hidden behind a benchmark name. Each example also includes a loss map that colours the training sequence token by token: green for tokens the model is trained to produce and grey for tokens provided only as context. This makes one important distinction visible. In an agentic trajectory, the tool response is context and receives no loss. In reinforcement learning, there may be no token-level training signal at all, only a reward assigned to the final result.

The loss map changes how each benchmark is interpreted. SWE-bench gives the model a repository and a bug report, but evaluates the patch it produces. The corresponding training shape is therefore code-editing data where the loss is applied to the generated patch. A tool-use benchmark evaluates the function name and arguments emitted by the model, so the matching training shape places loss on the generated function call or JSON. Once benchmarks are read at this level, the desired capability becomes a concrete list of training-data formats, which can then be translated directly into the mixture.

4. What actually exists to train on

A shopping list only works when the data actually exists. One common failure in mixture design is to assign a large share to a capability with very little real data, and then fill the remaining budget with whatever is available. To avoid this, each capability slot must be sized against the actual datasets that can feed it.

That sizing has to happen in two currencies: number of samples and number of tokens. They measure different things. A dataset may contain only a few thousand agentic trajectories but still occupy a large token budget because every trajectory contains many steps, tool calls and responses. A function-calling dataset may contain millions of samples while remaining comparatively small in tokens because each example is short.

A mixture designed only from sample counts will therefore misjudge how much training time each capability actually consumes. The token count determines the real weight of the dataset inside the run.

A dataset inventory for every capability slot, grouped and sortable by source, approximate samples, approximate tokens, license and the provenance tier from Session 3. A toggle switches between sample count and token count, and the ordering changes with it, making it clear why sample counts alone are misleading. Running totals show the real token supply available for each capability and feed back into the mixture composer. Violet scarcity markers flag where the Indic and agentic lanes run thin, identifying the gap that protected allocations and synthetic generation must cover.

This is where the scarcity of verified Indic data becomes measurable. Sorting by verified native tokens, rather than headline dataset size, shows how little genuinely native material exists. A mixture that assigns 25% of its budget to Indic languages cannot meet that target using verified sources alone.

That does not make the target invalid. It defines the amount of synthetic data that must be created. It also explains why the protected lane matters: the capabilities with the least supply are exactly the capabilities a naive mixture will starve first.

5. The training stages, and where reasoning enters

Before we discuss data ordering, we need a clear view of the full training lifecycle. The word training hides several different stages, each with a different objective, a different data format and a different scoring signal.

A model moves from random weights to pretraining, annealing, supervised fine-tuning, preference training and reinforcement learning before it becomes a finished assistant. The agentic and reasoning slots only make sense once we know which stage uses them, what the model is expected to produce and where the learning signal is applied. Keeping this sequence clear prevents the rest of the session from collapsing into one vague idea of training.

A training lifecycle timeline covering pretraining, mid-training annealing, supervised fine-tuning, reasoning training and preference alignment. Each stage shows its approximate token budget, making it clear that pretraining consumes the overwhelming majority of the data, while every later stage is comparatively small. Selecting a stage reveals its objective, the shape of one training example, the loss or reward signal and its typical scale. A live loss map changes from pretraining’s fully trained next-token sequence, to SFT’s response-only mask, and finally to reinforcement learning’s single reward signal. A marker shows exactly where reasoning training begins: after the base model already exists.

The reasoning stage is often placed incorrectly. Long reasoning traces are not simply mixed into pretraining and expected to produce a reasoning model. They are taught later. The model is first shown worked reasoning traces so it learns the structure of careful, multi-step problem solving. It is then trained using reinforcement learning with verifiable rewards, where the model attempts a problem, a checker evaluates the final answer, and that verdict updates the model without requiring a correct token-by-token target.

This sequence matters for the mixture we design now. The reasoning data reserved in Session 5 becomes the raw material for a much later training stage. A mixture decision made here therefore determines what will be possible when we reach reasoning training in Sessions 17 and 18.

6. The agentic slot, made concrete

Agentic data is the newest and least familiar capability slot, and it is also the one V4 had almost none of. The easiest way to understand it is to follow one complete task and observe the shape of the work.

Imagine a user asks the model to find every research grant in the United States relevant to a particular project, identify the people and laboratories that received those grants, and then determine which of those winners could plausibly buy a specific piece of hardware because their funded work requires exactly that kind of machine.

No single tool call can answer this. The model must plan the task, search for grants, read the results, launch follow-up searches based on what it finds, recover when a source is missing, try another route and maintain a running understanding of the entire investigation. Only after completing that chain can it produce the final answer.

This is what makes agentic data different. The training example is not a question followed by one response. It is a long trajectory of decisions, tool calls, observations, failures, recoveries and updated plans, all connected to one objective.

A step-by-step agentic training trajectory built around the same grant-research task. The sequence shows the model planning, emitting tool calls with arguments, reading observations, encountering a failed call, recovering and producing the final answer. Every token is coloured by whether it receives training loss: the model’s planning, tool calls and final response are green, while the user request and tool observations are grey. A counter tracks supervised tokens against context-only tokens, and a second view compares the full trajectory with a single function call to show the difference between basic tool use and a genuine multi-step agent.

The masking rule is critical. Tool observations are ground truth that the model may read and reason over, but must never be trained to reproduce. Applying loss to them would teach the model to invent tool results instead of calling the tool. These long trajectories are scarce, expensive and among the most valuable Tier A datasets available. They should therefore be protected for the annealing stage, rather than consumed early in pretraining. That allocation must be decided while composing the mixture because the data cannot be recovered once it has been spent.

7. The reasoning slot and the effort dial

The reasoning capability we want is not one fixed behaviour. A good assistant should answer an easy question quickly and spend serious effort only when the problem requires it. Modern systems expose this through a reasoning-effort control, where the caller can request low, medium, high or ultra reasoning depending on the difficulty of the task.

At inference time, this looks like a simple setting. The ability to obey that setting, however, must be learned during training. A model can only produce different reasoning depths if it has seen examples across that full range, from short single-step answers to long reasoning traces that explore alternatives, verify intermediate results and correct themselves before answering.

The control therefore does not create reasoning at inference time. It selects from reasoning behaviours the model was deliberately trained to perform.

The same problem solved at four reasoning levels: low, medium, high and ultra. Switching between them shows the reasoning trace grow from a short direct answer into a longer argument that checks intermediate steps and considers alternatives. The supervised reasoning tokens are shown in green, while counters track trace length, token usage and approximate accuracy as effort increases. A budget control connects the requested thinking budget to the length of reasoning produced. The panel also shows how the training mixture contains traces grouped by length, while the later reinforcement-learning stage teaches the model to respond consistently to the reasoning-effort setting.

The practical consequence is that reasoning cannot be treated as one uniform data slot. The mixture needs short, medium and long traces across mathematics, coding and general problem solving, otherwise the behaviour may remain tied to one domain or one fixed depth. Reserving a reasoning share therefore means reserving a distribution of trace lengths, while the curriculum determines the order in which the model sees them. Short reasoning provides the foundation; progressively longer traces teach the model to sustain, verify and correct a deeper line of thought.

8. Selecting the best data while the run is happening

So far, we have treated the mixture as a set of proportions fixed before training begins. That is no longer enough. Modern training systems can select the most useful data continuously while the run is in progress, instead of freezing the entire recipe at the start.

At any point in training, some candidate batches will teach the model far more than others. If we can estimate which batches contain the strongest learning signal for the model in its current state, we can prioritize them and avoid spending compute on low-value data. This allows the same token supply and compute budget to produce more useful learning.

This is the idea behind OPUS, the data-selection method used in our V4 production run and later formalized as a principled training technique.

A live view of OPUS selecting data one iteration at a time. Candidate batches arrive with domain tags, and each batch is scored by estimating how useful its training update would be against a stable proxy direction. The most useful fraction is retained in green, while the rest is rejected in red. A readout shows the effective-token multiplier and the small compute overhead introduced by selection. The critical control is the Always-On toggle. With protection disabled, an English-heavy proxy rejects most Indic and agentic batches, causing their contribution to fall toward zero. With protection enabled, a fixed share of every iteration is reserved for scarce capabilities regardless of their selector score.

Both sides of this design tension are valid. Aggressive selection uses compute more efficiently. In V4, OPUS retained only about 40% of the candidate data, delivered roughly a sixfold increase in effective token value, and added only a few percent of compute overhead. However, the selector defines usefulness through its proxy. An English-heavy proxy naturally undervalues native Indic text and unfamiliar agentic trajectories, starving the exact capabilities we want to build.

V4 solved this by placing Indic data in an always-on lane fixed at 8% of every batch, outside the selector’s control. V5 extends the same protection to Indic, agentic and reasoning data. The final design is therefore an aggressive selector operating above a protected capability floor.

9. Curriculum, or the order the model learns in

A mixture decides how much of each kind of data the model sees. A curriculum decides when it sees it. The order matters almost as much as the proportions. If the model encounters difficult material before it has the required foundations, much of that exposure is wasted. A flat mixture used from the first token to the last also misses the advantage of learning simpler patterns before harder ones.

Modern pretraining runs therefore move through deliberate stages. Training begins with broad general text to establish language, factual knowledge and basic structure. Once that foundation exists, the mixture shifts toward code, mathematics, science and reasoning-heavy data. Very long sequences are introduced later, after the model already knows how to read and reason, so that it can learn to preserve information across long contexts without having to learn every other capability at the same time.

Within each stage, the data also follows a difficulty ladder, beginning with simple material and progressing toward genuinely advanced examples. The curriculum is therefore not only a sequence of domains. It is a controlled progression across both capability and difficulty.

A curriculum timeline from the seed model through the general, reasoning, long-context and final annealing stages. Moving through the run updates a stacked mixture bar, showing general web gradually falling while code, reasoning and long-context data increase. A protected reserve of the strongest Tier A Indic and agentic data appears only near the end. A difficulty panel moves from the simplest examples to genuinely advanced material, with one concrete example at each level. The violet anneal marker makes one decision visible: the best data must be saved deliberately, not merely discovered at the end.

The final anneal is one of the highest-leverage stages in the entire plan. Most of pretraining runs on a broad mixture. Near the end, the learning rate is reduced and the model is trained for a short period on a small reserve of exceptionally high-quality data. Although this phase is small, the capability gain can be disproportionately large.

That gain is only possible if the reserve survives the main run. If the selector consumes the best Indic, agentic and reasoning data early, there is nothing special left for annealing. The final anneal therefore begins as a Session 5 data-allocation decision: identify the best material now, protect it from ordinary sampling and spend it only when the model is ready to benefit from it.

10. Keeping the run stable when the mixture moves

There is a serious risk inside every mixture transition. Whenever the data distribution changes, whether at a growth-stage boundary or at the beginning of the anneal, the gradients seen by the model also change. If that transition is too abrupt, training can become unstable: the gradient norm rises sharply, the loss spikes and the run may diverge.

V4 encountered this directly. A sudden increase in the Hindi share interacted with embeddings that had been frozen, causing the gradient norm to jump by roughly 150× over a short stretch. An event of that scale can destroy a training run if it is not detected and corrected quickly. Mixture transitions therefore cannot be treated as simple configuration changes. They must be introduced gradually and monitored as carefully as any major architectural change.

A live gradient-norm trace that advances with training steps. Applying a sudden mixture change produces an immediate spike, and enabling frozen embeddings makes the same transition far more violent. A warmup control then spreads the change across several thousand tokens and reduces the spike. A readout reports the spike multiplier, making the relationship between mixture shift, frozen parameters and training instability directly visible.

V4’s mitigation was simple: never change the mixture in one hard step. Every transition is blended across a warmup band of several billion tokens, allowing the model to move gradually from the old distribution to the new one.

This is also why the architecture and mixture are frozen before the main run begins. Changing the blend casually during live training can destroy days of expensive compute. The course schedule therefore treats mixture changes as planned, infrequent and carefully monitored transitions.

11. The assignment

From here the work is yours, and it is the most consequential design artifact the course has asked for so far. Each one of you will draft the mixture-and-curriculum plan for V5 as a written specification, and it has to be specific enough to defend. It states a share of the budget for every capability slot, and for the Indic slot it states the split across the verified, unverified, translated, and synthetic tiers rather than hiding behind a single headline number. It names the agentic and reasoning and long-context slots explicitly and points each one at the datasets from the inventory that will fill it. It fixes the protected always-on floor that the selector is not allowed to cross, it declares the anneal reserve that will be held back for the cooldown, and it lays out the difficulty and reasoning-length bands with a concrete example for each. And it commits to justifying these numbers through small proxy runs at the one-billion and three-billion scale before any of them is trusted at full scale, because the lesson that runs through this whole arc is that a data decision is a hypothesis until a cheap experiment has tested it. Alongside the plan the cleaning continues toward the cumulative target, now aimed at the slots the mixture shows to be starved.

Evaluation Stragtegy You will be evaluated on how well your plan would hold up if a reviewer sat across from you and pushed on every number, so the grade rests on the quality of your reasoning and the evidence behind your choices, and a tightly argued short plan will score well while padding earns nothing. A strong submission gives a defended share of the budget to every capability lane and states the Indic split across its verified, unverified, translated, and synthetic tiers, and it ties each lane back to the benchmarks it is meant to win so that a reviewer can see why each number is what it is. It sizes every lane against the real supply from the inventory and says plainly where a share can only be reached by repeating data or by generating it, and a plan that quietly hands a large share to a lane that has almost no real data behind it will lose marks for exactly the wishful accounting this session exists to prevent. It fixes the protected floor and declares the reserve it is holding back for the anneal, and it lays out the difficulty and reasoning-length bands with a real example at each level. The thing that separates a good plan from an excellent one is whether it is written as a testable hypothesis, so the highest marks go to the person who specifies a concrete proxy experiment at the one-billion or three-billion scale and name the metric that would confirm or refute their mixture, and the very highest go to the one that actually run that proxy and bring the numbers back. Your plan is reviewed the way a technique candidate is reviewed, in the open and against these criteria, and it is looked at only once your team has met the data-gating threshold, because a mixture is only as trustworthy as the cleaned and documented tokens standing behind it, and the plans that survive this review are the ones that will shape the mixture the whole cohort actually trains on.

Submission You're submitting link to your Github Repo README.md where we can see evaluate your work.

12. What this session commits us to and what comes next

Three commitments carry forward from this session.

First, the mixture is treated as the decision that makes the model. Its proportions, provenance tiers and curriculum are written down and defended. We begin from the capabilities we intend to demonstrate, map them to benchmarks, and let those benchmarks determine the data we need.

Second, scarce and valuable data is protected by design. Verified Indic data, agentic trajectories and long reasoning traces cannot be left entirely to a global selector that may starve them. They require protected floors, deliberate reservation and a place in the final anneal, where the model is ready to extract the most value from them.

Third, every proportion remains a hypothesis until tested. Before the mixture is allowed to shape the full run, it must survive one-billion and three-billion parameter proxy runs. The proxy results decide what remains, what changes and what is removed.

Session 6 takes this mixture and turns it into the physical data stream the training loop will consume. We will cover sharding, sequence packing, deterministic shuffling, pause-and-resume behaviour and dataloader throughput. Session 5 defines the recipe. Session 6 builds the system that can execute that recipe at scale.

The one thing to carry from this session

A mixture is a set of trade-offs made against a fixed token budget and composed backward from the capabilities we intend to win. Agentic and coding ability, controllable reasoning and native Indic fluency must each receive enough of the model’s attention. OPUS then selects the most useful tokens during training, while protected lanes preserve the scarce data that the selector would otherwise reject. Finally, the curriculum controls the order in which the model receives that data, so the capabilities grow without destabilizing the run.

Transcript

Video

Studio

GMeet

PREVIOUS

Session 4 - Data Cleaning and Deduplication

Mark Complete