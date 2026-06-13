import io
import librosa
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
import streamlit as st


@st.cache_resource
def load_voice_encoder():
    """
    Caches the Resemblyzer VoiceEncoder model in memory to prevent reload delays.
    """
    return VoiceEncoder()


def get_voice_embedding(audio_bytes):
    """
    Extracts a 256-dimensional embedding vector from raw audio bytes.
    """
    try:
        encoder = load_voice_encoder()

        # Load audio from binary data stream at 16kHz sampling rate
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        wav = preprocess_wav(audio)
        embedding = encoder.embed_utterance(wav)
        return embedding.tolist()
    except Exception as e:
        st.error(f"Voice embedding extraction error: {e}")
        return None


def identify_speaker(new_embedding, candidates_dict, threshold=0.65):
    """
    Compares a new embedding vector against a dictionary of saved embeddings.
    """
    if new_embedding is None or not candidates_dict:
        return None, 0.0

    best_sid = None
    best_score = -1.0

    # Ensure the query vector is a flat NumPy array
    new_emb_np = np.array(new_embedding).flatten()

    for sid, stored_embedding in candidates_dict.items():
        if stored_embedding is not None:
            # Safely cast candidate vectors to NumPy arrays
            stored_emb_np = np.array(stored_embedding).flatten()

            # Resemblyzer embeddings are normalized, so dot product gives cosine similarity
            similarity = np.dot(new_emb_np, stored_emb_np)
            if similarity > best_score:
                best_score = similarity
                best_sid = sid

    if best_score >= threshold:
        return best_sid, float(best_score)

    return None, float(best_score)


def verify_student_voice(audio_bytes, enrolled_student_ids, candidates_dict, threshold=0.65):
    """
    Processes audio bytes and matches them against the provided enrolled student records.
    """
    if not audio_bytes or not enrolled_student_ids or not candidates_dict:
        return False, None

    try:
        # Step 1: Extract embedding vector from the audio clip
        new_embedding = get_voice_embedding(audio_bytes)
        if new_embedding is None:
            return False, None

        # Step 2: Filter the candidate database to check ONLY the targeted student IDs
        filtered_candidates = {
            sid: candidates_dict[sid]
            for sid in enrolled_student_ids
            if sid in candidates_dict
        }

        # Step 3: Run the vector similarity identification check
        matched_id, score = identify_speaker(new_embedding, filtered_candidates, threshold)

        if matched_id:
            return True, matched_id

        return False, None
    except Exception as e:
        st.error(f"Voice verification sequence failed: {e}")
        return False, None


def process_bulk_audio(audio_bytes, candidates_dict, threshold=0.65):
    """
    Splits long continuous audio files into smaller talking fragments and identifies speakers.
    """
    try:
        encoder = load_voice_encoder()

        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        
        # Split audio stream into non-silent segments (top_db=30 handles average background hums)
        segments = librosa.effects.split(audio, top_db=30)

        identified_results = {}

        for start, end in segments:
            # Ignore ultra-short audio pops and click artifacts under 0.5 seconds
            if (end - start) < sr * 0.5:
                continue

            segment_audio = audio[start:end]
            wav = preprocess_wav(segment_audio)
            embedding = encoder.embed_utterance(wav)

            # Match individual chunk snippet
            sid, score = identify_speaker(embedding, candidates_dict, threshold)

            if sid:
                # Save or update the highest match match score for the user
                if sid not in identified_results or score > identified_results[sid]:
                    identified_results[sid] = float(score)

        return identified_results
    except Exception as e:
        st.error(f"Bulk audio compilation split error: {e}")
        return {}
