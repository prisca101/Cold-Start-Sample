import streamlit as st
import joblib
import numpy as np
from lightfm import LightFM
import pandas as pd
from lightfm.data import Dataset
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as sparse_vstack
import scipy.sparse as sp
import gspread
from gspread.exceptions import SpreadsheetNotFound, GSpreadException
from google.oauth2.service_account import Credentials
from datetime import datetime

# =====================
# CUSTOM STYLES
# =====================
st. set_page_config(layout="wide")

# Add Google Fonts
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=Merriweather:wght@400;700&display=swap" rel="stylesheet">',
    unsafe_allow_html=True
)

# =====================
# APP FUNCTIONALITY
# =====================

# Load assets
@st.cache_resource
def load_assets():
    # Load model with custom handler
    model = joblib.load('lightfm_model.pkl')    
    # Load other data
    data = joblib.load('supporting_data.pkl')
    return model, data

model, data = load_assets()

# Access the features
user_features_test = data['user_features_test']
books_df = data['books_df']
users_df = data['users_df']
user_id_mapping = data['user_id_mapping']
item_id_mapping = data['item_id_mapping']
cold_user_ids = data['cold_user_ids']
test_ratings = data['test_ratings']
num_users = len(user_id_mapping)



def get_star_rating(rating):
    if rating == 5 or rating == 0:
        return 3
    else:
        return round(rating / 2)
    

@st.dialog("Book Details", width="large")
def show_book_details(isbn):
    book = books_df[books_df['ISBN'] == isbn].iloc[0]
    st.image(book['Image-URL-L'] if pd.notna(book['Image-URL-L']) else "https://placehold.co/150x200?text=Cover+Not+Available", use_container_width=True)
    st.subheader(book['Cleaned_Title'])
    st.markdown(f"**Author:** {book['Book-Author']}")
    
    genres = book['genres']
    if isinstance(genres, list):
        genres = ", ".join(genres)
    st.markdown(f"**Genres:** {genres}")

    # Add more detailed info if needed
    st.markdown("---")
    st.markdown("More book details can go here...")


# Define dialog function for book details
@st.dialog("Book Description", width="large")
def show_book_details_dialog(isbn):
    book_info = books_df[books_df['ISBN'] == isbn].iloc[0]
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if pd.notna(book_info.get('Image-URL-L')):
            st.image(book_info['Image-URL-L'], use_container_width=True)
        else:
            st.image("https://placehold.co/300x400?text=Cover+Not+Available", 
                     use_container_width=True)
    
    with col2:
        # Custom styled title (larger font + teal color)
        st.markdown(
            f"<h1 style='font-size: 30px; color: #52c3be;'>{book_info['Cleaned_Title']}</h1>", 
            unsafe_allow_html=True
        )
        
        # Custom styled author (light gray color)
        st.markdown(
            f"<div style='color: #ebefe7; font-size: 22px; margin-bottom: 12px;'>by {book_info['Book-Author']} (Author)</div>", 
            unsafe_allow_html=True
        )
        
        # Genres
        genres = book_info['genres']
        if isinstance(genres, list):
            genre_pills = "".join(
                [f'<span class="genre-pill">{g}</span>' 
                for g in genres[:3]]
            )
            st.markdown(f'<div class="genre-container">{genre_pills}</div>', 
                        unsafe_allow_html=True)
        
        # Series
        if book_info['Series'] and book_info['Series'] != "Standalone":
            st.markdown(
                f"""
                <div style='font-style: italic;'>
                    Part of the {book_info['Series']} series
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("  \n")

        # Description
        st.markdown("### Description")
        if book_info.get('description') and not pd.isna(book_info['description']):
            st.markdown(book_info['description'])
        else:
            st.markdown("*No description available.*")

        st.markdown("---")
        st.markdown(f"Published in {int(book_info['Year-Of-Publication'])} by {book_info['Publisher']}")



# =====================
# APP LAYOUT
# =====================

st.image("Book Recommender System.png", use_container_width=True)

st.markdown("  \n")
st.markdown("  \n")
st.markdown("  \n")

st.markdown("""
    <style>
        .hero {
            background-color: #fac826;
            padding: 2rem;
            text-align: center;
            width: 100%;
            margin: 20px 0;
            border-radius: 20px;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .hero-title {
            font-size: 3rem;
            font-weight: 700;
            color: #3d3a3a !important;
            font-family: 'Playfair Display', serif;
            margin: 0;
        }
        .custom-divider {
            width: 100px;
            height: 4px;
            background-color: #3d3a3a;
            margin: 20px auto;
            border-radius: 2px;
        }
        .hero-description {
            max-width: 700px;
            margin: 0 auto;
            font-size: 1.1rem;
            color: #3d3a3a;
            font-family: 'Arial', sans-serif;
        }
    </style>

    <div class="hero">
        <div class="custom-divider"></div>
        <p class="hero-description">
            Our collection comes from the BookCrossing Community, featuring thousands of books 
            from the golden era of late 20th century literature. While we can't offer the latest bestsellers, 
            we hope to help you discover forgotten masterpieces and hidden gems.
        </p>
    </div>
""", unsafe_allow_html=True)

st.image("dots.png", use_container_width=True) 



st.markdown("""
            <style>
                .book-grid-title {
                    text-align: center;
                    font-size: 1.4rem !important;
                    font-weight: 700 !important;
                    line-height: 1.3;
                    margin: 8px 0 4px 0 !important;
                    color: #52c3be;
                }
                .book-grid-author {
                    text-align: center;
                    font-size: 1rem !important;
                    font-style: italic;
                    margin: 0 0 8px 0 !important;
                    color: #ebefe7;
                }
                .genre-pill {
                    display: inline-block;
                    background-color: transparent !important; /* Transparent background */
                    color: #ebefe7 !important;              /* Text color */
                    border: 1px solid #275445 !important;   /* Border with same color */
                    padding: 3px 10px !important;
                    border-radius: 16px !important;
                    font-size: 0.8rem !important;
                    margin: 2px 4px 2px 0 !important;
                    white-space: nowrap !important;
                    transition: all 0.2s ease !important;   /* Smooth transitions */
                }

                .genre-pill:hover {
                    background-color: rgba(218, 225, 174, 0.1) !important; /* Subtle hover effect */
                    transform: translateY(-1px);
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                }

                .genre-container {
                    text-align: center;
                    margin: 8px 0 12px 0 !important;
                    line-height: 1.5 !important;
                }
            </style>
            """, unsafe_allow_html=True)

if 'overlap' not in st.session_state:
    st.session_state.overlap = set()
if 'actual_isbns' not in st.session_state:
    st.session_state.actual_isbns = []
if 'actual_books' not in st.session_state:
    st.session_state.actual_books = []
if 'rating_dict' not in st.session_state:
    st.session_state.rating_dict = {}




# =====================
# PREFERENCES SECTION 
# =====================
with st.container():
    left_spacer, content, right_spacer = st.columns([0.05, 0.30, 0.05])
    with content:   
        st.image("banner_choose_pref.png", use_container_width=True) 

        st.markdown("  \n")
        st.markdown("  \n")
        st.markdown("  \n")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            selected_user = st.selectbox(
                "Select a Cold-Start User Sample:",
                cold_user_ids[70:81]
            )

        user_data = users_df[users_df['User-ID'] == selected_user].iloc[0]

        st.markdown("  \n")
        st.markdown("  \n")

        # col1, col2 = st.columns(2)
        # with col1:
        #     st.markdown(
        #         f"""
        #         <div style='text-align: center; font-size: 1.5rem; font-weight: bold;'>
        #             Top 3 Genres
        #         </div>
        #         """, unsafe_allow_html=True)

        #     genres = user_data['fav_genres']
        #     if isinstance(genres, list):
        #         genre_pills = "".join(
        #             [f'<span class="genre-pill">{g}</span>' 
        #             for g in genres[:3]]
        #         )
        #         st.markdown(f'<div class="genre-container">{genre_pills}</div>', 
        #                     unsafe_allow_html=True)

        # with col2:
        st.markdown(
            f"""
            <div style='text-align: center; font-size: 1.5rem; font-weight: bold;'>
                Top 3 Authors
            </div>
            """, unsafe_allow_html=True)
        
        authors = user_data["fav_authors"]
        if isinstance(authors, list):
            author_pills = "".join(
                [f'<span class="genre-pill">{a}</span>' 
                for a in authors[:3]]
            )
            st.markdown(f'<div class="genre-container">{author_pills}</div>', 
                        unsafe_allow_html=True)




# =====================
# RECOMMENDATION CONTROLS
# ===================== 
left_spacer, content, right_spacer = st.columns(3) 
with content:
    st.markdown("  \n")
    st.markdown("  \n")

    num_recommendations = 10



    # Generate recommendations button
    if st.button("𝐆𝐞𝐧𝐞𝐫𝐚𝐭𝐞 𝐑𝐞𝐜𝐨𝐦𝐦𝐞𝐧𝐝𝐚𝐭𝐢𝐨𝐧𝐬 💕", type="primary", use_container_width=True):
        user_internal_id = user_id_mapping[selected_user]
        current_user_features = user_features_test[user_internal_id]
        # Generate predictions
        scores = model.predict(
            user_ids=user_internal_id,
            item_ids=np.arange(len(item_id_mapping)),
            user_features=user_features_test,
            num_threads=4
        )

        # Get recommendations
        top_indices = np.argsort(-scores)[:num_recommendations]
        isbn_list = list(item_id_mapping.keys())
        recommended_isbns = [isbn_list[idx] for idx in top_indices]

        # Get actual interactions from test set and create a rating dictionary
        actual_books = test_ratings[test_ratings['User-ID'] == selected_user]
        actual_isbns = actual_books['ISBN'].tolist()

        # Create a dictionary for ISBN to rating mapping
        rating_dict = dict(zip(actual_books['ISBN'], actual_books['Book-Rating']))

        overlap = set(recommended_isbns) & set(actual_isbns)

        # Store recommendations in session state
        st.session_state.recommended_isbns = recommended_isbns
        st.session_state.show_recommendations = True
        st.session_state.overlap = overlap
        st.session_state.actual_isbns = actual_isbns
        st.session_state.rating_dict = rating_dict  # Store the rating dictionary

    st.metric("Recommendation Precision", 
    f"{len(st.session_state.overlap)}/{len(st.session_state.actual_isbns)} matches",
    help="Number of recommended books that user actually interacted with in the test set")



# Check if we have recommendations to show
if "show_recommendations" in st.session_state and st.session_state.show_recommendations:
    
    st.image("banner_rec.png", use_container_width=True) 
 
    st.markdown("  \n")
    st.markdown("  \n")
    st.markdown("  \n")
    st.markdown("  \n")
    st.markdown("  \n")

    # Create two main columns
    # Recommendations by Model section
    st.markdown(
        f"""
        <div style='text-align: center; font-size: 1.5rem; font-weight: bold;'>
            Recommendations by Model
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")

    # Create grid for model recommendations (4 columns)
    rec_rows = [st.session_state.recommended_isbns[i:i+4] for i in range(0, len(st.session_state.recommended_isbns), 4)]
    for row in rec_rows:
        cols = st.columns(4)
        for col_index, isbn in enumerate(row):
            book_info = books_df[books_df['ISBN'] == isbn].iloc[0]
            match_indicator = " ✅" if isbn in st.session_state.overlap else ""
            
            with cols[col_index]:
                with st.container():
                    # Book cover
                    if pd.notna(book_info.get('Image-URL-L')):
                        st.markdown(
                            f"""
                            <div style='text-align: center;'>
                                <img src="{book_info['Image-URL-L']}" width="150">
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown(
                            """
                            <div style='text-align: center;'>
                                <img src="https://placehold.co/150x200?text=Cover+Not+Available" width="150">
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Title
                    title = book_info["Cleaned_Title"] + match_indicator
                    if len(title) > 50:
                        title = title[:47] + "..."
                    st.markdown(f'<div class="book-grid-title">{title}</div>', unsafe_allow_html=True)
                    
                    # Author
                    author = book_info["Book-Author"]
                    if len(author) > 30:
                        author = author[:27] + "..."
                    st.markdown(f'<div class="book-grid-author">by {author}</div>', unsafe_allow_html=True)
                    
                    # Genres
                    genres = book_info['genres']
                    if isinstance(genres, list):
                        genre_pills = "".join(
                            [f'<span class="genre-pill">{g}</span>' 
                            for g in genres[:2]]
                        )
                        st.markdown(f'<div class="genre-container">{genre_pills}</div>', 
                                    unsafe_allow_html=True)
                    
                    # Details button
                    btn_left, btn_center, btn_right = st.columns([1, 2, 1])
                    with btn_center:
                        if st.button("View Details", key=f"rec_detail_{isbn}", use_container_width=True):
                            st.session_state.selected_book = isbn

                    st.markdown("  \n")
                    st.markdown("  \n")

    # Actual Interactions section
    st.markdown("  \n")
    st.markdown("  \n")
    st.markdown(
        f"""
        <div style='text-align: center; font-size: 1.5rem; font-weight: bold; margin-top: 2rem;'>
            Actual Interactions (Test Set)
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")

    if len(st.session_state.actual_isbns) > 0:
        # Create grid for actual interactions (4 columns)
        actual_rows = [st.session_state.actual_isbns[i:i+4] for i in range(0, len(st.session_state.actual_isbns), 4)]
        for row in actual_rows:
            cols = st.columns(4)
            for col_index, isbn in enumerate(row):
                if col_index >= len(row):  # Skip if no book in this position
                    continue
                    
                book_info = books_df[books_df['ISBN'] == isbn].iloc[0]
                
                # Get rating from dictionary
                rating = st.session_state.rating_dict.get(isbn, 0)
                if rating == 0:
                    rating = 5
                
                with cols[col_index]:
                    with st.container():
                        # Book cover
                        if pd.notna(book_info.get('Image-URL-L')):
                            st.markdown(
                                f"""
                                <div style='text-align: center;'>
                                    <img src="{book_info['Image-URL-L']}" width="150">
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown(
                                """
                                <div style='text-align: center;'>
                                    <img src="https://placehold.co/150x200?text=Cover+Not+Available" width="150">
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Title
                        title = book_info["Cleaned_Title"]
                        if len(title) > 50:
                            title = title[:47] + "..."
                        st.markdown(f'<div class="book-grid-title">{title}</div>', unsafe_allow_html=True)
                        
                        # Author
                        author = book_info["Book-Author"]
                        if len(author) > 30:
                            author = author[:27] + "..."
                        st.markdown(f'<div class="book-grid-author">by {author}</div>', unsafe_allow_html=True)
                        
                        # Rating
                        stars = "⭐" * get_star_rating(rating)
                        st.markdown(f'<div style="text-align: center; margin: 5px 0;">{stars} ({rating}/10)</div>', unsafe_allow_html=True)
                        
                        # Genres
                        genres = book_info['genres']
                        if isinstance(genres, list):
                            genre_pills = "".join(
                                [f'<span class="genre-pill">{g}</span>' 
                                for g in genres[:2]]
                            )
                            st.markdown(f'<div class="genre-container">{genre_pills}</div>', 
                                        unsafe_allow_html=True)
                        
                        # Details button
                        btn_left, btn_center, btn_right = st.columns([1, 2, 1])
                        with btn_center:
                            if st.button("View Details", key=f"actual_detail_{isbn}", use_container_width=True):
                                st.session_state.selected_book = isbn

                        st.markdown("  \n")
                        st.markdown("  \n")
    else:
        st.warning("No recorded interactions for this user in test set")



# Show book details page if a book is selected
if "selected_book" in st.session_state and st.session_state.selected_book:
    show_book_details_dialog(st.session_state.selected_book)
