import streamlit as st
from src.ui.base_layout import style_background_dashboard, style_base_layout
from src.components.header import header_dashboard
from PIL import Image
import numpy as np
from src.pipelines.face_pipeline import predict_attendance, get_face_embeddings, train_classifier
from src.pipelines.voice_pipeline import get_voice_embedding, verify_student_voice
from src.database.db import get_all_students, create_student, get_student_subjects, get_student_attendance, unenroll_student_to_subject
import time
from src.components.dialog_enroll import enroll_dialog
from src.components.subject_card import subject_card


def student_dashboard():
    student_data = st.session_state.student_data
    student_id = student_data['student_id']
    
    st.header('Your Enrolled Subjects')
    st.divider()
    
    subjects = get_student_subjects(student_id)
    cols = st.columns(2)
    
    for i, sub_node in enumerate(subjects):
        sub = sub_node['subjects']
        sid = sub['subject_id']
        sname = sub['name']
        scode = sub['subject_code']
        
        with cols[i % 2]:
            subject_card(
                name=sname,
                code=scode,
                section=sub.get('section', 'A'),
                stats=[('📅', 'Total', 12), ('✅', 'Attended', 9)]
            )
            
            with st.expander("🎙️ Audio Attendance Session", expanded=False):
                st.write("Click record and say clearly: **'I am present'**")
                
                voice_file = st.audio_input(label="Tap to speak", key=f"voice_mic_{sid}_{i}")
                
                if voice_file:
                    with st.spinner("AI is matching voice print..."):
                        audio_bytes = voice_file.read()
                        
                        candidates_dict = {student_id: student_data['name']}
                        is_match, matched_id = verify_student_voice(audio_bytes, candidates_dict)
                        
                        if is_match and matched_id == student_id:
                            st.success(f"Voice Verified! Attendance marked for {sname} ✅")
                            st.balloons()
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Voice signature match mismatch! Try again.")


def student_screen():
    style_background_dashboard()
    style_base_layout()

    if "student_data" in st.session_state:
        student_dashboard()
        return

    c1, c2 = st.columns(2, vertical_alignment='center', gap='xxlarge')
    with c1:
        header_dashboard()
    with c2:
        if st.button("Go back to Home", type='secondary', key='loginbackbtn', shortcut="control+backspace"):
            st.session_state['login_type'] = None
            st.rerun()

    tab1, tab2 = st.tabs(["📷 FaceID Login", "🎙️ VoiceID Login"])

    if "show_reg" not in st.session_state:
        st.session_state.show_reg = False
    if "reg_image" not in st.session_state:
        st.session_state.reg_image = None

    with tab1:
        st.header('Login using FaceID', text_alignment='center')

        cam_col, upload_col = st.columns(2)
        photo_source = None

        with cam_col:
            st.subheader("Option A: Capture Live")
            cam_image = st.camera_input("Position your face in the center", key="face_cam_input")
            if cam_image:
                photo_source = cam_image

        with upload_col:
            st.subheader("Option B: Upload Image")
            uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="face_upload_input")
            if uploaded_image:
                photo_source = uploaded_image

        if photo_source:
            img = np.array(Image.open(photo_source))
            with st.spinner('AI is scanning your face..'):
                detected, all_ids, num_faces = predict_attendance(img)
                if num_faces == 0:
                    st.warning('Face not found!')
                elif num_faces > 1:
                    st.warning('Multiple faces found')
                else:
                    if detected:
                        keys_list = list(detected.keys())
                        if keys_list:
                            student_id = keys_list[0]
                            all_students = get_all_students()
                            student = next((s for s in all_students if str(s['student_id']) == str(student_id)), None)

                            if student:
                                st.session_state.is_logged_in = True
                                st.session_state.user_role = 'student'
                                st.session_state.student_data = student
                                st.toast(f"Welcome Back {student['name']} (Verified via Face)")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"Face matched with ID {student_id}, but no matching profile found in Database!")
                    else:
                        st.info('Face not recognized! You might be a new student!')
                        st.session_state.show_reg = True
                        st.session_state.reg_image = img

    with tab2:
        st.header('Login using VoiceID', text_alignment='center')
        st.write("Click below to record your verification passphrase.")

        audio_source = st.audio_input("Record your voice sample", key="voice_audio_input")

        if audio_source:
            voice_detected = False
            detected_student_id = None

            with st.spinner('AI is processing your voice signature..'):
                audio_bytes = audio_source.read()

                try:
                    st.info("🔄 AI आवाज की जांच कर raha hai... कृपया रुकें...")
                    result = get_voice_embedding(audio_bytes)

                    if isinstance(result, (tuple, list)):
                        if len(result) >= 2:
                            voice_detected = result[0]
                            detected_student_id = result[1]
                        elif len(result) == 1:
                            voice_detected = True if result[0] else False
                            detected_student_id = result[0]
                    else:
                        voice_detected = True if result else False
                        detected_student_id = result

                except Exception as e:
                    st.error("❌ An error occurred inside get_voice_embedding calculation:")
                    st.exception(e)

                if voice_detected and detected_student_id and detected_student_id != "Unknown":
                    all_students = get_all_students()
                    student = next((s for s in all_students if str(s['student_id']) == str(detected_student_id)), None)

                    if student:
                        st.session_state.is_logged_in = True
                        st.session_state.user_role = 'student'
                        st.session_state.student_data = student
                        st.toast(f"Welcome Back {student['name']} (Verified via Voice)")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Voice recognized but no student profile matched ID: {detected_student_id}")
                else:
                    st.error("Voice pattern match nahi hua! Please try again or use FaceID.")

    if st.session_state.show_reg and st.session_state.reg_image is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.header('Register new Profile')
            new_name = st.text_input("Enter your name to register", placeholder='E.g. Hamza Rizvi')

            if st.button("Complete Registration & Login", type="primary", use_container_width=True):
                if new_name.strip() == "":
                    st.warning("Please enter a valid name first!")
                else:
                    with st.spinner("Creating profile and training AI face models..."):
                        try:
                            new_student_id = f"STU_{int(time.time())}"

                            try:
                                new_student = create_student(new_student_id, new_name.strip())
                            except TypeError:
                                try:
                                    new_student = create_student(id=new_student_id, name=new_name.strip())
                                except TypeError:
                                    new_student = {"student_id": new_student_id, "name": new_name.strip()}

                            if not new_student or not isinstance(new_student, dict):
                                new_student = {"student_id": new_student_id, "name": new_name.strip()}

                            face_embeddings = get_face_embeddings(st.session_state.reg_image)
                            train_classifier(new_student_id, face_embeddings)

                            st.session_state.is_logged_in = True
                            st.session_state.user_role = 'student'
                            st.session_state.student_data = new_student

                            st.session_state.show_reg = False
                            st.session_state.reg_image = None

                            st.toast(f"Registration successful! Welcome {new_student['name']}")
                            time.sleep(1)
                            st.rerun()

                        except Exception as e:
                            st.error("❌ An error occurred during registration:")
                            st.exception(e)
                            st.session_state.show_reg = False
                            st.session_state.reg_image = None