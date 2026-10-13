hotpotqa_few_shots = [
"""Question: Were Scott Derrickson and Ed Wood of the same nationality?

Available paragraphs (10 titles; use Search[title] to read each):
- Scott Derrickson
- Ed Wood
- Adam Collis
- Doris Wishman
- Conrad Brooks
- Sandra Bullock
- Steven Spielberg
- Tim Burton
- Quentin Tarantino
- Wes Anderson

Thought 1: I need the nationalities of Scott Derrickson and Ed Wood. Read the Scott Derrickson paragraph first.
Action 1: Search[Scott Derrickson]
Observation 1: [Scott Derrickson]: Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.
Thought 2: Scott Derrickson is American. Now read the Ed Wood paragraph.
Action 2: Search[Ed Wood]
Observation 2: [Ed Wood]: Edward Davis Wood Jr. (October 10, 1924 - December 10, 1978) was an American filmmaker.
Thought 3: Ed Wood is also American. Both share the same nationality.
Action 3: Finish[yes]""",

"""Question: The Oberoi family is part of a hotel company that has a head office in what city?

Available paragraphs (10 titles; use Search[title] to read each):
- The Oberoi family
- The Oberoi Group
- Marriott International
- Hyatt
- Hilton Worldwide
- Wyndham Hotels & Resorts
- Tata Group
- Reliance Industries
- ITC Limited
- Indian Hotels Company

Thought 1: First find which hotel company the Oberoi family is part of.
Action 1: Search[The Oberoi family]
Observation 1: [The Oberoi family]: The Oberoi family is an Indian family that is famous for its involvement in hotels through The Oberoi Group.
Thought 2: The family is part of The Oberoi Group. Now find the Group's head-office city.
Action 2: Search[The Oberoi Group]
Observation 2: [The Oberoi Group]: The Oberoi Group is a hotel company with its head office in Delhi.
Action 3: Finish[Delhi]""",

"""Question: What award did the director of the film "The Hurt Locker" win for that film?

Available paragraphs (10 titles; use Search[title] to read each):
- The Hurt Locker
- Kathryn Bigelow
- Mark Boal
- Zero Dark Thirty
- Point Break
- James Cameron
- Avatar (2009 film)
- Academy Award for Best Director
- 82nd Academy Awards
- Jeremy Renner

Thought 1: First find who directed "The Hurt Locker".
Action 1: Search[The Hurt Locker]
Observation 1: [The Hurt Locker]: The Hurt Locker is a 2008 American war thriller film directed by Kathryn Bigelow.
Thought 2: Directed by Kathryn Bigelow. Now find what award she won for that film.
Action 2: Search[Kathryn Bigelow]
Observation 2: [Kathryn Bigelow]: Kathryn Ann Bigelow won the Academy Award for Best Director for The Hurt Locker (2009).
Action 3: Finish[Academy Award for Best Director]"""
]


hotpotqa_solver_system_prompt = """
Solve a multi-hop question answering task in the HotpotQA Distractor setting. You will be shown a question and a list of 10 paragraph TITLES (2 contain the supporting facts, 8 are distractors). The paragraph CONTENTS are hidden — you must Search a title to read its body, which is returned as the next observation.

You must answer using ONLY the information in the paragraphs you Search. Do NOT rely on outside knowledge or memorised facts.

Solve the task with interleaving Thought / Action / Observation steps:
- `Thought N: <reasoning>` — plan which paragraph to read next, or connect facts you have read. N is the step number.
- `Action N: Search[<title>]` — read the paragraph with that title. The next observation will be its body. The title should match one of the 10 shown (case-insensitive; a clear substring also works).
- `Action N: Finish[<answer>]` — submit the final answer and end the task.
  - For yes/no questions, answer with "yes" or "no".
  - For factoid questions, provide a concise span (a name, number, date, or short phrase) found in the paragraphs.

CRITICAL output rules:
1. Output EXACTLY ONE step per turn: either a single `Thought N: ...`, a single `Action N: Search[<title>]`, or a single `Action N: Finish[<answer>]`. Do not chain multiple steps.
2. Reasoning MUST be prefixed with `Thought N: `. Action lines MUST be prefixed with `Action N: ` so step counts stay aligned. Do NOT emit bare reasoning or unlabeled actions.
3. Do NOT use <think> or </think> tags.
4. Typical questions need 2-3 Searches plus a handful of Thoughts. As soon as you have the supporting evidence, issue Finish — the task fails if you never Finish.
"""
