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

**The question this project asks:** how small can the model you own be, and still get most
of the way to the big rented one?

An analogy: the large model is an expensive consultant who's brilliant immediately but bills
every phone call. The small model is a junior hire who's useless on day one -- but you can
train them, and afterwards they work for free, forever. The question isn't just whether
training works. It's *how junior you can go* -- how small a hire still does the job well
enough -- because the smaller they are, the cheaper they are to keep.

## What we did

**1. Collected the raw material.** ~1550 real job postings pulled from the French national
employment service.

**2. Created an answer key.** 100 postings were corrected *by hand*, field by field, to say
exactly what the correct table row should be. This is the yardstick -- without it, there's
no way to tell whether the model is right, only whether it sounds confident. Building it
took the largest share of the effort, and it's the part most projects skip.

**3. Trained the juniors.** 881 postings became lessons. The large model produced a
first-pass table for each, those were cleaned up, and the small models studied them
repeatedly -- like a trainee working through hundreds of solved exercises.

**4. Graded everyone against the same answer key**, at two different sizes, so the result is
a *curve* rather than a single number. All marked on the same 100 hand-corrected postings.

## What we found

| | size | score | share of the big model's quality |
|---|---|---|---|
| Small model, untrained | 1.5 billion | 0.49 | 53% |
| Small model, trained | 1.5 billion | 0.83 | 91% |
| **Bigger small model, trained** | **3 billion** | **0.88** | **96%** |
| Large rented model | ~70 billion | 0.92 | 100% |

Reading the score: 1.00 would be a perfect match with the hand-corrected answer key.

Two things stand out, and the second is the more useful one.

**Training does the heavy lifting.** It takes the small model from 53% to 91% of the big
one's quality -- on a free graphics card, in a few hours.

**Then it stops paying.** Doubling the model from 1.5 to 3 billion adds only another 5
points, and the last 4% didn't move at all: not for 43% more training data, not for two
different attempts at fixing the training examples. The gains come fast and then flatten,
which means **you don't need a big model here -- 3 billion settings, 23 times smaller than
the rented one, gets you 96% of the way.**

The obvious next step -- a 7-billion model -- doesn't fit on the free hardware. So 3 billion
is the honest end of this particular curve, not a claim about where it would flatten with a
bigger budget.

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
training examples both failed, in opposite directions, and more data didn't help either.

The one thing that *did* move it was making the model bigger. That's the clearest evidence in
the project for where the size limit actually bites: this column isn't a labelling problem
you can fix with better examples, it's a judgement the small model can't reliably make. It's
also about half of the entire remaining gap -- so if you wanted the last 4%, this is the one
thing you'd have to buy a bigger model for.

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

## The answer, and why it's useful

**It does not match the big model.** The remaining 4% is real, not measurement noise -- we
re-ran the training three times to check exactly that -- and it's mostly the skills column.

But that wasn't quite the question. The question was how small you can go, and the answer is
concrete: **3 billion settings, 23 times smaller, running free on hardware you control,
reaching 96% of a rented model's quality on cheaper requests.**

Whether that's a good trade depends entirely on the use. For filtering, first-pass sorting,
or anything a human reviews anyway, 96% at zero marginal cost is obviously worth it. For
something where the last 4% matters and nobody checks the output, it isn't.

Knowing *which* case you're in, with numbers behind it, is the actual deliverable.
