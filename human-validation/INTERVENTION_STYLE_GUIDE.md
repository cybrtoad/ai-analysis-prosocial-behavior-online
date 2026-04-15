# Intervention Style Coding Guide

## Overview

For each record, code the **intervention text** (the moderator's action) into one of five style categories. This is the only place in the rating process where you are evaluating the moderator's behavior rather than the user's response.

Code the **primary** style. Most interventions have a clear dominant style. If the intervention genuinely combines two styles equally, use the decision rules at the end of this document.

---

## The Five Categories

### punitive

**Definition**: The intervention focuses on punishment, censure, or consequences. The moderator communicates that the user has violated a rule and that a consequence has been or will be applied. The emphasis is on the violation and its cost, not on explaining or redirecting.

**Key signals**:
- Explicit statement of a rule violation ("You have violated rule X")
- Account warning, suspension, or ban notice
- Content removal or revert with minimal or no explanation
- Language emphasizing what the user did wrong rather than what they should do instead
- Escalating language ("This is your final warning")

**Examples**:
> *"Your comment has been removed for violating our civility policy. Further violations will result in a ban."*
> *"This is a warning. Personal attacks are not permitted here."*
> *"Reverted. Vandalism is not tolerated."*
> *[Account suspension notice with no explanation of what specifically was wrong]*

**NOT punitive if**: The moderator explains *why* something is a violation or *how* to fix it. Explanation shifts it toward educative.

---

### educative

**Definition**: The intervention explains the rules, norms, or reasoning behind a policy. The moderator is teaching — providing context that helps the user understand *why* their behavior was problematic and what is expected instead.

**Key signals**:
- Explains the reasoning behind a rule, not just the rule itself
- Quotes or paraphrases a policy with explanation of its purpose
- Walks through what was wrong about the specific content and why
- Provides links to guidelines or documentation with encouragement to read them
- Tone is instructional rather than punitive

**Examples**:
> *"CMV requires that you engage with the points in a delta-worthy post, not just restate your position. The reason is that the subreddit is specifically for exploring whether your view can be changed, so the focus should be on the other person's argument."*
> *"Wikipedia has a neutral point of view policy, which means all significant viewpoints on a topic should be represented, not just the majority view. Your edit removed the minority perspective entirely, which is why it was reverted."*
> *"On StackExchange, answers should directly address the question asked. Your post contains useful background but doesn't answer the specific problem the user described."*

**NOT educative if**: The moderator states the rule but gives no explanation of why it exists or how it applies to this specific situation. That is closer to punitive.

---

### suggestive

**Definition**: The intervention redirects by offering a concrete alternative approach, reframing the situation, or proposing a specific path forward. Rather than focusing on what was wrong, the moderator focuses on what would be better.

**Key signals**:
- Offers a specific alternative ("Instead of X, try Y")
- Proposes an edit, revision, or different approach
- Reframes the situation to open a constructive path
- Asks a guiding question that redirects without judging
- Uses language like "consider," "what if," "you might try," "have you thought about"

**Examples**:
> *"Instead of reverting the entire section, could you add a [citation needed] tag and start a Talk page discussion? That way the issue gets flagged without removing the content entirely."*
> *"It sounds like you're frustrated with the other user. What if you tried restating your argument without the personal comments? The underlying point is worth engaging with."*
> *"Your answer would be stronger if you included a working code example. As written, it's hard for the asker to apply this to their specific situation."*

**NOT suggestive if**: The moderator offers only vague encouragement ("please be more constructive") without a concrete alternative. That is closer to educative (if it explains why) or punitive (if it just states the problem).

---

### positive

**Definition**: The intervention focuses on affirming and encouraging prosocial behavior. The moderator emphasizes what the user is doing right, expresses appreciation, or frames the feedback in a way that leads with positive reinforcement.

**Key signals**:
- Explicitly acknowledges good behavior or contributions
- Welcome messages to new users
- Delta awards with explanation of what the user did well
- Framing feedback as "keep doing X" rather than "stop doing Y"
- Warm, affirming tone with no punitive element

**Examples**:
> *"Welcome to ChangeMyView! Your post is a great example of what we're looking for here — you've described your view clearly and you're genuinely open to being persuaded. Looking forward to the discussion."*
> *"∆ Awarded. You provided a specific counter-example that the OP hadn't considered, which is exactly the kind of argument this subreddit is designed for."*
> *"This is a really thorough edit — you've properly cited the new content and your edit summary explains the change clearly. This is exactly the kind of contribution Wikipedia needs."*

**NOT positive if**: The intervention leads with positive framing but then includes a substantive warning or correction. Code based on the primary emphasis. A "sandwich" (positive-negative-positive) that centers a warning is closer to educative or punitive.

---

### structural

**Definition**: The intervention is a platform-level action that does not involve direct communication with the user. The moderator has taken a structural action (removing content, closing a thread, applying a template, blocking an account) and the action itself is the intervention, without any accompanying explanation or message.

**Key signals**:
- Content removal with no accompanying message
- Thread lock or closure
- Bot-generated standard notices with no personalization
- Block/ban with no explanatory message
- AutoModerator actions

**Examples**:
> *[Post removed — no message]*
> *[Thread closed]*
> *"Your account has been blocked." [automated template with no specific explanation]*
> *[AutoModerator: "Your post has been removed because your account is less than 30 days old."]*

**NOT structural if**: There is any personalized explanation, suggestion, or acknowledgment attached to the structural action. The explanatory component dominates.

---

## Decision Rules for Ambiguous Cases

### When two styles seem equally present

Use this priority order: if the intervention includes a structural action AND any explanatory text, code the explanatory text's primary style (not structural). If the intervention is half punitive and half educative, code educative — explanation takes precedence over sanction when both are present.

Priority: **positive > suggestive > educative > punitive > structural**

In practice: code the "higher" style unless the lower style clearly dominates.

### Common ambiguous pairs

**punitive vs. educative**: Does the moderator explain *why* the rule exists or how it applies to this specific content? If yes → educative. If the intervention only names the rule and states the consequence → punitive.

**educative vs. suggestive**: Is the emphasis on *explaining the problem* or on *proposing a path forward*? If the concrete alternative is the centerpiece → suggestive. If the explanation of the norm is the centerpiece → educative.

**suggestive vs. positive**: Does the intervention offer an alternative (suggestive) or affirm existing good behavior (positive)? An invitation to do something different is suggestive; praise for what the user already did is positive.

**positive vs. educative with positive framing**: If the moderator's primary purpose is to teach or correct but uses a warm tone, code it educative. The warmth of delivery does not make it positive — positive is reserved for interventions where affirmation IS the primary purpose.

### When you genuinely cannot decide

Write the two candidate codes in the "notes" column of the rating form (e.g., "educative/suggestive") and flag it. These will be discussed in the adjudication meeting.

---

## Quick Reference Card

| Style | Core question | Key phrase examples |
|---|---|---|
| punitive | "What consequence are you applying?" | "warning," "removed," "violation," "banned," "final warning" |
| educative | "What are you teaching?" | "the reason is," "this policy means," "on this platform," "guidelines say" |
| suggestive | "What alternative are you offering?" | "instead of," "consider," "what if," "you might try," "have you thought about" |
| positive | "What are you affirming?" | "welcome," "great contribution," "delta," "exactly right," "thank you for" |
| structural | "Is there any personalized text?" | [no text / bot template / removal notice only] |
