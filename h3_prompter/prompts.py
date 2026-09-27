"""System prompt + request builder for MiniMax H3 Full-Reference (R2V) prompts.

The system prompt is STATIC on purpose: llama-server caches the KV of an identical prefix,
so every run after the first skips re-processing it (faster time-to-first-token).
All per-run data (duration, assets, task) goes into the user message.
"""

from __future__ import annotations

import re

R2V_FIELDS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)

TASKS = {
    "auto": None,
    "reference generation": "[reference generation]",
    "image edit": "[keyframe completion]",
    "video editing": "[video editing]",
    "video continuation": "[video continuation]",
    "keyframe completion": "[keyframe completion]",
}

FRAME_ANCHORS = [
    "none",
    "reference 1 = first frame",
    "last reference = last frame",
    "reference 1 = first frame + last reference = last frame",
]

SYSTEM_PROMPT_R2V = """You are an expert prompt writer for MiniMax H3, an audiovisual video generation model, in Full-Reference mode. You convert a user's request plus labeled reference assets into ONE final H3 prompt that follows the official MiniMax H3 Full-Reference prompt-writing guide.

OUTPUT FORMAT - exactly these six sections, in this order, each header on its own line followed by its content:
subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:

Output only the prompt. No preamble, no markdown, no code fences, no notes after the last section.

TASK TYPES (summary prefix; combine with " + ", e.g. [keyframe completion + reference generation])
- [keyframe completion]: an image is a concrete frame anchor (first frame, keyframe, last frame, edited keyframe).
- [reference generation]: an asset guides identity/appearance/style/environment without being a concrete frame.
- [video editing]: an existing source video is directly modified.
- [video continuation]: new content continues, extends, resumes or transitions from a source video.
- [audio reuse]: the same audio signal is reused in full or in part.
- [audio reference]: only music style, voice timbre or dialogue characteristics are referenced.

LABELS
- <Picture N> = supplied image, <Video N> = supplied video, <Audio N> = supplied audio. Use exactly the labels listed in the request; never invent assets. Each category is numbered independently; numbers do not encode pairings between categories.
- <Subject N> = a reusable visible element (person, animal, object, outfit, environment, style) taken from the assets.

subject_definitions
- One line per subject: "<Subject N> is the <what> in <Picture N>, with <key visible attributes>." When the same subject comes from several assets, combine the sources and state what each asset provides.
- A picture that is itself a frame anchor stays a standalone <Picture N>: "<Picture N> is the first frame of [Shot 1], showing ..." (or keyframe / last frame of [Shot N]).
- A source video for editing: "<Video 1> is the source video for the target video edit." A source for continuation: "<Video 1> is the source video that the target video continues."
- One line per audio asset, e.g. "<Audio 1> is the voice-timbre reference for <Subject 1> (S1), containing ..." or "<Audio 1> is the synchronized audio track of <Video 1> and is reused in the target video."

summary
- One paragraph that starts with the task-type prefix. For video editing, continue right after the prefix with: "The target video is an edited version of <Video 1>." Then state what the target video shows and how each reference is used.

retention_analysis
- One line per subject: "<Subject N> (appears in [Shot X], [Shot Y]): <marker> - <what is kept and what changes>."
- One line per frame-anchor picture: "<Picture N> ([Shot 1] first frame): fully_preserved - ..." (or "([Shot N] last frame)", "([Shot N] keyframe)").
- Video entries: e.g. "<Video 1> (motion, camera and timing): fully_preserved - ..." or "<Video 1> (cut and pacing structure): weak_reference - ...".
- Visual markers: fully_preserved, partially_preserved, attribute_transfer, weak_reference.
- Audio markers: fully_copy (exact reuse, lips sync to it), partially_copy, reference (timbre/style only), weak_reference. Use the marker given in the request for each audio asset.
- For edits, everything the user did not ask to change is preserved; name exactly what is kept and what changes.
- Marker choice for edits: a subject whose identity stays but whose clothing/hair/color changes -> partially_preserved. attribute_transfer only when an attribute is taken from ANOTHER reference asset (e.g. the dress in <Picture 1> put on <Subject 1> from <Video 1>).

detailed_description
- First sentence: visual style (e.g. live-action cinematic, realistic sitcom, 3D animation, anime) and lighting.
- Shots in playback order: "[Shot 1] ..." with no timestamp, later cuts "[Shot N] At MM:SS.mmm, the shot cuts to ...". Timestamps strictly increase and stay below the target duration. Prefer few shots (about one per 3-5 seconds); one continuous shot is fine for short clips.
- Frame anchors in natural phrasing: "the shot begins from <Picture 1>", "the shot's keyframe corresponds to <Picture 2>", "the shot ends on <Picture 3>". These phrases are only for pictures, never for <Video N>.
- Video editing / continuation: describe the COMPLETE resulting video, not only the change; cite <Video N> naturally where its source state, structure or continuation applies. Newly added actions, backgrounds or plot elements are legitimate additions.
- Describe what is actually visible in the supplied frames: the real setting, props, colors, lighting direction, the subject's hair and features, each action in order with approximate timing, and the real camera behavior (e.g. "a static medium shot", "the camera slowly pushes in"). Never hedge ("whether static or moving", "any visible text", "if present") and never write editing-process or meta language ("unchanged from the source", "frame by frame", "no cuts added", "without any alteration"): write the final video as if describing it to someone who has not seen the source.
- Keep framing statements consistent across the prompt (do not call the subject off-center in one sentence and centered in the next).
- When the request is brief (e.g. "white dress"), make the new element concrete and plausible: cut, length, sleeves, neckline, fabric, how it moves and catches the light.
- Insert each label at a subject's first appearance in every shot and briefly repeat key attributes ("<Subject 2>, the woman in the red coat from Shot 1").
- Camera motion as natural action inside the sentence with type + amplitude + speed, e.g. "the camera pushes in with small amplitude at slow speed". Vocabulary: push in, pull out, zoom in/out, pan left/right, truck left/right, tilt up/down, pedestal up/down, arc shot, tracking shot, static shot, shakes slightly/strongly, POV, roll clockwise/counterclockwise.
- Speakers get stable IDs (S1), (S2) reused everywhere, with a short voice description at their first line. Speech: <d>[Language] exact words</d>, e.g. <d>[English] Hey! Watch your dog!</d> or <d>[Indonesian] Ayo cepat!</d>; the square brackets are literal. Keep the user's dialogue verbatim - never translate, paraphrase, shorten or merge it. Do not invent dialogue unless the request allows it. After a line, state that the speaker closes the mouth or continues an action.
- Visible on-screen text in English double quotes, verbatim.
- Concrete, visible actions and physical detail; no abstract mood words. Speech must fit the time (about 2.5 words per second).

AUDIO ASSETS - decide each <Audio N>'s use from the request and write the matching marker and task prefix:
- "use this exact sound / lip-sync to it / keep the original audio" -> fully_copy, [audio reuse]; only part of it -> partially_copy.
- "voice like this / this voice timbre / music in this style" -> reference, [audio reference]; barely relevant -> weak_reference.
- No <Audio N> supplied -> H3 generates all sound itself: never claim that the source audio is kept, copied or "unchanged", and give <Video N> no audio retention; just describe the sound in overall_soundscape.
- If the request says nothing: the synchronized audio track of a video that is being edited or continued -> fully_copy (the original sound is kept); a standalone audio clip -> reference (voice timbre of the speaker it belongs to).

overall_soundscape
- 1-4 sentences: ambience, action sounds, non-verbal human sounds. Never repeat dialogue or describe music here.

non_diegetic_music
- 1-3 sentences: instruments, tempo, rhythm, dynamics - no emotional adjectives.
- Follow the request: music asked for -> describe it; no music / silence asked for -> N/A; nothing said -> a short score only if it clearly suits the scene, otherwise N/A.

LANGUAGE
- The user may write in any language (often Indonesian). Write every section in English, except dialogue/lyrics inside <d>...</d>.

LENGTH
- Follow the requested detailed_description length; other sections stay short. Dialogue-dense scenes prioritize fitting the complete spoken timeline over word count."""


def build_user_text(
    *,
    instruction: str,
    task: str,
    frame_anchor: str,
    duration_s: float,
    frames: int,
    pictures: list[str],
    videos: list[str],
    audios: list[str],
    length: str,
    allow_invented_dialogue: bool,
    asset_notes: str,
    extra_rules: str,
) -> str:
    lines = ["REQUEST", instruction.strip() or "(no instruction - infer a natural, coherent use of the assets)"]
    if asset_notes.strip():
        lines += ["", "ASSET ROLES (from the user)", asset_notes.strip()]
    lines += ["", "TARGET"]
    lines.append(
        f"- Target video duration: {duration_s:.2f} seconds ({frames} frames at 24 fps). "
        f"All [Shot N] timestamps must be below {duration_s:.2f} s."
    )
    tags = []
    if TASKS.get(task):
        tags.append(TASKS[task])
    n_pics = len(pictures)
    last_pic = f"<Picture {n_pics}>" if n_pics else None
    if frame_anchor != "none" and n_pics:
        if "[keyframe completion]" not in tags:
            tags.insert(0, "[keyframe completion]")
        if frame_anchor.startswith("reference 1 = first frame"):
            lines.append(
                "- FRAME ANCHOR: <Picture 1> is the exact FIRST FRAME of [Shot 1] (standalone <Picture 1>, not a subject). "
                "Write in subject_definitions \"<Picture 1> is the first frame of [Shot 1], showing ...\", in retention_analysis "
                "\"<Picture 1> ([Shot 1] first frame): fully_preserved - ...\", and start [Shot 1] with \"the shot begins from <Picture 1>\". "
                "The opening framing, subjects, layout and lighting must match <Picture 1>."
            )
            if n_pics > 1:
                lines.append("- The other pictures are references (subjects/style/environment) unless stated otherwise.")
        if frame_anchor.endswith("last reference = last frame"):
            if frame_anchor.startswith("last reference") or n_pics > 1:
                lines.append(
                    f"- FRAME ANCHOR: {last_pic} is the exact LAST FRAME of the final shot: define it as "
                    f"\"{last_pic} is the last frame of [Shot N], showing ...\", retention \"{last_pic} ([Shot N] last frame): "
                    f"fully_preserved - ...\", and end the final shot with \"the shot ends on {last_pic}\" (N = final shot number)."
                )
    if task == "image edit" and n_pics:
        if frame_anchor.startswith("reference 1 = first frame"):
            lines.append(
                "- IMAGE EDIT: the video starts exactly on the unedited <Picture 1>; the requested change then happens "
                "visibly on screen (a natural transformation), while everything the user did not ask to change stays "
                "identical. Describe the change as an on-screen event with its timing."
            )
        else:
            lines.append(
                "- IMAGE EDIT: <Picture 1> is the edited keyframe - the target video shows <Picture 1> WITH the requested "
                "change already applied from the first frame. Define it as a standalone <Picture 1> (\"<Picture 1> is the "
                "edited keyframe of [Shot 1], showing ...\"), keep everything the user did not ask to change (identity, "
                "framing, layout, lighting), mark it partially_preserved in retention_analysis naming exactly what changes, "
                "then animate the edited scene naturally."
            )
        if "[keyframe completion]" not in tags:
            tags.insert(0, "[keyframe completion]")
    if task == "video editing":
        lines.append(
            "- VIDEO EDIT: <Video 1> is the source video. The summary must continue after the prefix with "
            "\"The target video is an edited version of <Video 1>.\" Preserve the source motion, camera, timing and anything "
            "not mentioned in the request; describe the complete resulting video shot by shot, including the edit."
        )
        lines.append(
            "- AUDIO FOR VIDEO EDIT: the final video keeps the ORIGINAL audio of <Video 1> (it is put back afterwards), "
            "so H3's sound is not used. Keep overall_soundscape to ONE short sentence matching the source ambience, write "
            "non_diegetic_music as N/A unless the user explicitly asks for music, and add no new speech. If the subject "
            "talks in the source, describe the mouth moving as natural speech in the same rhythm as in <Video 1> (no <d> "
            "text unless the user wrote the exact words), so the lips stay in sync with the original audio."
        )
    if task == "video continuation":
        lines.append(
            "- VIDEO CONTINUATION: the target video starts where <Video 1> ends (same subjects, place, lighting, momentum) "
            "and continues it for the target duration."
        )
    if tags:
        lines.append("- Task prefix: [" + " + ".join(t.strip("[]") for t in tags) + "]"
                     + " (append audio reuse / audio reference when audio assets are used).")
    else:
        lines.append("- Task prefix: choose the correct task type(s) from the request and the assets.")
    lines.append(
        {
            "compact": "- Length: compact - detailed_description about 150-250 words, fewest shots possible.",
            "standard": "- Length: standard - detailed_description about 350-500 words (official guideline).",
            "detailed": "- Length: detailed - detailed_description about 500-700 words.",
        }.get(length, "- Length: standard - detailed_description about 350-500 words (official guideline).")
    )
    lines.append(
        "- Dialogue: you may write short natural lines if they fit the scene."
        if allow_invented_dialogue
        else "- Dialogue: only use speech the user wrote (or speech kept from a source video/audio); otherwise no speech."
    )
    lines += ["", "ASSETS (use exactly these labels, connected in this order to the H3 node)"]
    if not (pictures or videos or audios):
        lines.append("- none")
    lines.extend(f"- {p}" for p in pictures)
    lines.extend(f"- {v}" for v in videos)
    lines.extend(f"- {a}" for a in audios)
    if extra_rules.strip():
        lines += ["", "EXTRA RULES", extra_rules.strip()]
    lines += ["", "Write the final MiniMax H3 prompt now."]
    return "\n".join(lines)


# ------------------------------------------------------------------ light post-processing

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
_SHOT1_TS = re.compile(r"\[Shot 1\]\s*At\s+00:00(?:\.0+)?\s*,?\s*", re.IGNORECASE)
_BAD_LANG = re.compile(r"<d>\s*(English|Indonesian|Chinese|Japanese|Korean|Spanish|French|German)\s+(?!\])")
_SHOT_TS = re.compile(r"(\[Shot \d+\]\s*)at\s+(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\b", re.IGNORECASE)


def _fix_ts(m: re.Match) -> str:
    prefix, mm, ss, ms = m.group(1), m.group(2), m.group(3), (m.group(4) or "0")
    return f"{prefix}At {int(mm):02d}:{ss}.{ms.ljust(3, '0')}"


def clean_output(text: str) -> str:
    t = text.strip()
    t = _FENCE.sub("", t).strip()
    # drop anything before the first section header (preambles)
    idx = t.find("subject_definitions:")
    if idx > 0:
        t = t[idx:]
    t = _SHOT1_TS.sub("[Shot 1] ", t)
    t = _BAD_LANG.sub(lambda m: f"<d>[{m.group(1)}] ", t)
    t = _SHOT_TS.sub(_fix_ts, t)
    # one blank line between sections
    for f in R2V_FIELDS[1:]:
        t = re.sub(r"\s*\n\s*" + f + r":", "\n\n" + f + ":", t)
    return t.strip()
