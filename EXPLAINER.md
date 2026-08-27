# What this project is, in plain language

## The problem

Job postings are written for humans. A site like France Travail has hundreds of thousands of
them, each a wall of free text: the company somewhere in the header, the contract length
buried mid-paragraph, the required skills scattered through bullet points, and half the
details simply not mentioned at all.

If you want to *do* anything with them at scale -- match candidates to jobs, chart which
skills employers actually ask for, filter by contract length -- you first need to turn that
prose into a table. Title here, company there, duration, location, skills, salary. Same
columns every time, and an honest blank when the posting genuinely doesn't say.

Doing that by hand is impossible at volume. Doing it with ordinary keyword rules fails
because every posting is worded differently.

## The two ways to solve it

Modern language models can read a posting and fill in the table. There are two options, and
they trade off against each other.

**Option 1: rent a very large model.** These are the systems behind ChatGPT-style products.
They're excellent out of the box -- you describe the table you want and they mostly fill it
in correctly. But you're renting: you pay per request, you send your data to someone else's
servers, you're subject to their rate limits and price changes, and the model is far too big
to run on your own hardware.

**Option 2: use a small model you own.** These run on a single graphics card -- even a free
one. They're private, free to run, and yours permanently. The catch is that out of the box
they're *bad* at this. Ask a small model to fill in the table and it produces a mess.

**The question this project asks:** if you take a small model and *train* it on examples of
the job done correctly, how much of that gap can you close?

An analogy: the large model is an expensive consultant who's brilliant immediately but bills
every phone call. The small model is a junior hire who's useless on day one -- but you can
train them, and afterwards they work for free, forever. The question is how good the junior
gets.

## What we did

**1. Collected the raw material.** ~770 real job postings pulled from the French national
employment service.

**2. Created an answer key.** 100 postings were corrected *by hand*, field by field, to say
exactly what the correct table row should be. This is the yardstick -- without it, there's
no way to tell whether the model is right, only whether it sounds confident. Building it
took the largest share of the effort, and it's the part most projects skip.

**3. Trained the junior.** The remaining ~650 postings became lessons. The large model
produced a first-pass table for each, those were cleaned up, and the small model studied
them repeatedly -- like a trainee working through hundreds of solved exercises.

**4. Graded everyone against the same answer key.** Three contestants: the large rented
model, the small model *before* training, and the small model *after* training. All marked
on the same 100 hand-corrected postings.

## What we found

| | size | score |
|---|---|---|
| Large rented model | ~70 billion settings | **0.92** |
| Small model, untrained | 1.5 billion | 0.49 |
| **Small model, after training** | 1.5 billion | **0.83** |

Reading the score: 1.00 would be a perfect match with the hand-corrected answer key. Roughly,
the trained small model gets **about 88% of individual fields right**.

**Training closed about three quarters of the gap** -- using a model roughly **47 times
smaller**, on a free graphics card, in a few hours.

It also turned out **cheaper per request than expected**. A rented model needs the whole
table format explained to it every single time, which costs money proportional to text
length. A trained model has already internalised the format, so its instructions are about a
third as long. We measured this rather than assumed it: the shorter instructions cost
**nothing** in accuracy.

And on two specific columns -- contract type and contract duration -- **the small trained
model beat the large rented one**. Not because it's cleverer, but because the training
examples encoded a rule the big model doesn't follow: when a posting says "12 to 24 months",
record 12. The student learned a convention its teacher never had.

## What didn't work, and why that matters

The trained model is still clearly worse at **listing required skills**. It over-lists --
naming skills the posting never mentions. Two separate attempts to fix this by adjusting the
training examples both failed, in opposite directions. The evidence points at a genuine
capability limit rather than a fixable data problem: at this size, with this many examples,
the model can't reliably judge which skills belong.

We also discovered two problems in our *own* method and reported both:

- **About 10% of the exam questions had appeared in the study material.** The employment
  service re-lists the same job under new reference numbers with small edits, so our
  duplicate check missed them. This inflates the trained model's score specifically. We
  measured the effect and report the clean number alongside.
- **Re-running the same training twice gives different results** -- by enough to swamp
  several of the improvements we were chasing. So some earlier conclusions were withdrawn as
  "too small to distinguish from random variation."

Both of these make the headline number *less* impressive. They're in the report because a
result you can't trust isn't worth having.

## Why the answer is useful either way

The original hypothesis was that a small trained model could **match** the large rented one.
It doesn't -- the remaining gap is real, not noise, and the report says so.

But the practical finding stands: **a model 47 times smaller, running free on hardware you
control, reaches about 90% of the quality of a rented one, with cheaper requests.** For
plenty of real uses -- filtering, first-pass sorting, anything where a human reviews the
output anyway -- that trade is clearly worth making. For uses needing top accuracy, it isn't.

Knowing *which* case you're in, with numbers behind it, is the actual deliverable.
