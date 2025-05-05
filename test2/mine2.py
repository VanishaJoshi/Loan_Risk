import streamlit as st

# Set up session state to persist data between steps
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}

def next_step():
    st.session_state.step += 1

def prev_step():
    st.session_state.step -= 1

st.title("🎓 Student Admission Form")

if st.session_state.step == 1:
    st.header("Step 1: Personal Information")
    with st.form("personal_info"):
        name = st.text_input("Full Name")
        dob = st.date_input("Date of Birth")
        gender = st.radio("Gender", ["Male", "Female", "Other"])
        phone = st.text_input("Phone Number")
        next_button = st.form_submit_button("Next")
        if next_button:
            st.session_state.form_data.update({
                "Name": name,
                "Date of Birth": str(dob),
                "Gender": gender,
                "Phone": phone
            })
            next_step()

elif st.session_state.step == 2:
    st.header("Step 2: Academic Information")
    with st.form("academic_info"):
        grade = st.selectbox("Applying for Grade", ["1", "2", "3", "4", "5", "6"])
        previous_school = st.text_input("Previous School Name")
        achievements = st.text_area("Academic Achievements")
        prev_button = st.form_submit_button("Back")
        next_button = st.form_submit_button("Next")
        if prev_button:
            prev_step()
        if next_button:
            st.session_state.form_data.update({
                "Grade Applied": grade,
                "Previous School": previous_school,
                "Achievements": achievements
            })
            next_step()

elif st.session_state.step == 3:
    st.header("Step 3: Parent/Guardian Information")
    with st.form("guardian_info"):
        guardian_name = st.text_input("Parent/Guardian Name")
        guardian_contact = st.text_input("Contact Number")
        address = st.text_area("Home Address")
        prev_button = st.form_submit_button("Back")
        submit_button = st.form_submit_button("Submit")
        if prev_button:
            prev_step()
        if submit_button:
            st.session_state.form_data.update({
                "Guardian Name": guardian_name,
                "Guardian Contact": guardian_contact,
                "Address": address
            })
            st.success("🎉 Form Submitted Successfully!")
            st.write("Here is the information you submitted:")
            st.json(st.session_state.form_data)
            st.balloons()
            st.session_state.step = 1  # Reset form for new entry
