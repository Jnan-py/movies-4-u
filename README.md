# Movies 4 U

Movies 4 U is a Streamlit-based movie recommendation system that allows users to search for movies, view top-rated films, get recommendations based on a selected movie, and even explore user-based recommendations (for study purposes). The app leverages data from CSV files (including movie posters, overviews, genres, and keywords) to deliver recommendations using cosine similarity computed on movie features.

## Features

- **Search Movie:**

  - Search for a movie by name and view details such as the poster, summary, and genre information.
  - Get a "See Also" section with recommendations for similar movies.
  - Access additional information via a link to the movie's Wikipedia page.

- **Top Movies:**

  - Display top movies based on rating.
  - Adjustable slider to select the number of movies to display.
  - View movie posters and summaries for each top movie.

- **Movie Based Recommender:**

  - Select a movie and receive recommendations for similar movies based on genre, keywords, and overview using cosine similarity.

- **User Based Recommendations (Study Purpose):**
  - Explore a user-based recommendation system (using the "ml-100K" dataset) that demonstrates collaborative filtering techniques.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Jnan-py/movies-4-u.git
   cd movies-4-u
   ```

2. **Create a Virtual Environment (Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Application:**

   ```bash
   streamlit run movie.py
   ```

2. **Navigation:**

   - Use the sidebar to select between different pages:
     - **Search Movie:** Look up a movie by name and see its details along with similar recommendations.
     - **Top Movies:** View top-rated movies based on user ratings.
     - **Movie Based:** Get movie recommendations based on a selected movie.
     - **User Based (study purpose):** Explore user-based recommendations (collaborative filtering) using the "ml-100K" dataset.

3. **Movie Details:**
   - For each movie, the app displays a poster, a summary, genres, keywords, and provides a link for more information (e.g., Wikipedia).

## Project Structure

```
movies-4-u/
│
├── movie.py                   # Main Streamlit application
├── datasets/                # Folder containing movie data CSV files and other related datasets
│   ├── 10000 Movies Data/   # CSV files with movie details, overviews, genres, etc.
│   └── movie_poster.csv     # CSV mapping movie IDs to poster URLs
├── Pictures/                # Folder containing fallback images (e.g., Image-not-found.png)
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies
```

## Technologies Used

- **Streamlit:** Interactive web application framework.
- **Pandas & NumPy:** Data manipulation and numerical operations.
- **Scikit-learn:** Feature extraction and similarity calculations using CountVectorizer and cosine similarity.
- **Requests:** For fetching images from URLs.
- **Pillow (PIL):** For image processing.
- **Matplotlib & Plotly:** For data visualization.
- **Streamlit Option Menu:** For an intuitive sidebar navigation interface.
- **Urllib:** For URL encoding and generating Wikipedia links.

---

Save these files in your project directory. To run the application, activate your virtual environment and run:

```bash
streamlit run movie.py
```

Feel free to modify the documentation as needed. Enjoy recommending movies with Movies 4 U!
