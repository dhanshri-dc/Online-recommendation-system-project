# =========================================
# ONLINE COURSE RECOMMENDATION SYSTEM
# =========================================

# =========================
# IMPORT LIBRARIES
# =========================

import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# PAGE TITLE
# =========================

st.title("Online Course Recommendation System")

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("online_course_recommendation.csv")

# =========================
# BINARY ENCODING
# =========================

df['certification_offered'] = df[
    'certification_offered'
].map({
    'Yes': 1,
    'No': 0
})

df['study_material_available'] = df[
    'study_material_available'
].map({
    'Yes': 1,
    'No': 0
})

# =========================
# LABEL ENCODING
# =========================

le = LabelEncoder()

df['difficulty_level'] = le.fit_transform(
    df['difficulty_level']
)

# =========================
# SAMPLE DATA
# =========================

sample_df = df.sample(
    5000,
    random_state=42
)

sample_df = sample_df.reset_index(drop=True)

# =========================
# FEATURE SELECTION
# =========================

features = sample_df[[
    'course_duration_hours',
    'certification_offered',
    'difficulty_level',
    'rating',
    'enrollment_numbers',
    'course_price',
    'feedback_score',
    'study_material_available',
    'time_spent_hours',
    'previous_courses_taken'
]]

# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

scaled_features = scaler.fit_transform(
    features
)

# =========================
# CONTENT-BASED SIMILARITY
# =========================

similarity = cosine_similarity(
    scaled_features
)

# =========================
# COLLABORATIVE FILTERING
# =========================

user_course_matrix = df.pivot_table(
    index='user_id',
    columns='course_name',
    values='rating'
).fillna(0)

course_similarity = cosine_similarity(
    user_course_matrix.T
)

# =========================
# CONTENT-BASED FUNCTION
# =========================

def recommend_courses(course_name):

    course_index = sample_df[
        sample_df['course_name'] == course_name
    ].index[0]

    similarity_scores = list(
        enumerate(similarity[course_index])
    )

    sorted_courses = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommended_courses = []

    for i in sorted_courses[1:6]:

        recommended_courses.append(
            sample_df.iloc[i[0]]['course_name']
        )

    return recommended_courses

# =========================
# COLLABORATIVE FUNCTION
# =========================

def recommend_collaborative(course_name):

    course_index = user_course_matrix.columns.get_loc(
        course_name
    )

    similarity_scores = list(
        enumerate(course_similarity[course_index])
    )

    sorted_courses = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommended_courses = []

    for i in sorted_courses[1:6]:

        recommended_courses.append(
            user_course_matrix.columns[i[0]]
        )

    return recommended_courses

# =========================
# HYBRID FUNCTION
# =========================

def hybrid_recommendation(course_name):

    content_rec = recommend_courses(
        course_name
    )

    collaborative_rec = recommend_collaborative(
        course_name
    )

    final_rec = list(
        set(content_rec + collaborative_rec)
    )

    return final_rec[:5]

# =========================
# MODEL SELECTION
# =========================

model_option = st.selectbox(
    "Select Recommendation Model",
    [
        "Content-Based",
        "Collaborative",
        "Hybrid"
    ]
)

# =========================
# COURSE SELECTION
# =========================

selected_course = st.selectbox(
    "Select Course",
    sample_df['course_name'].unique()
)

# =========================
# BUTTON
# =========================

if st.button("Recommend Courses"):

    # CONTENT-BASED

    if model_option == "Content-Based":

        recommendations = recommend_courses(
            selected_course
        )

    # COLLABORATIVE

    elif model_option == "Collaborative":

        recommendations = recommend_collaborative(
            selected_course
        )

    # HYBRID

    else:

        recommendations = hybrid_recommendation(
            selected_course
        )

    # OUTPUT

    st.subheader("Recommended Courses")

    for course in recommendations:

        st.write(course)