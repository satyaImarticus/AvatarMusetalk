Where every file must go (also fully commented at the top of the script itself):

File	Location
avatar_ws_server.py	/workspace/MuseTalk/avatar_ws_server.py — must sit directly in the repo root, not /workspace/, not a subfolder
in_interviewer.mp4 (avatar video)	/workspace/MuseTalk/data/video/in_interviewer.mp4 — exact name, exact path
Model weights	/workspace/MuseTalk/models/... (from runpod_setup.sh)
Prepared avatar cache	/workspace/MuseTalk/results/v15/avatars/interviewer/ — built once by the preparation pass
Audio	Nothing to place — it's never a file on disk. It streams live from Gemini through Node → RunPod per-turn, and the server's own temp WAVs (/tmp/avatar_turns/) get deleted automatically right after each turn
