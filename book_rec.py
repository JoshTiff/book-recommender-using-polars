import gzip
import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer #type: ignore[import-untyped]
from sklearn.metrics.pairwise import cosine_similarity #type: ignore[import-untyped]
import polars as pl
import argparse
from pathlib import Path

class BookRecommender():
    def __init__(self, book_data: str, book_map: str, interactions_data: str, directory: argparse.Namespace):
        # Extract necessary data fields from each line of goodreads_books.json
        # Helper function for extracting data from books file
        def parse_fields(book_line: bytes) -> dict:
            data = json.loads(book_line)
            return {
            "book_id": data["book_id"],
            "title": data["title_without_series"],
            "ratings": data["ratings_count"],
            "url": data["url"],
            }

        # Store all books with over 25 reviews for use in search
        with gzip.open(f"{directory}/{book_data}", "r") as book_file:
            books_titles = []
            while book_line := book_file.readline():
                fields = parse_fields(book_line)
                
                try:
                    ratings = int(fields["ratings"])
                except ValueError: 
                    continue
                
                if ratings > 25: 
                    books_titles.append(fields)

        # Creates a polars DataFrame with book ids, titles, ratings, and normalized titles
        self.titles = pl.from_dicts(books_titles, schema = {"book_id": pl.String, "title": pl.String, "ratings": pl.Int64, "url": pl.String})
        self.titles = self.titles.with_columns(self.titles["title"].str.replace_all("[^a-zA-Z0-9 ]", "").str.to_lowercase().alias("norm_titles"))
        
        # Creates polars DataFrame from interactions file with user ids, book ids, and ratings
        self.interactions_df = pl.read_csv(f"{directory}/{interactions_data}", columns = ["user_id", "book_id", "rating"], new_columns = ["user_id", "csv_id", "rating"])
        self.interactions_df = self.interactions_df.cast({"user_id": pl.String, "csv_id": pl.String})

        # Creates maps for book ids
        self.interactions_to_book_data_map = {}
        self.book_data_to_interactions_map = {}
        with open(f"{directory}/{book_map}", "r") as map_file:
            while map_line := map_file.readline():
                csv_id, book_id = map_line.strip().split(",")
                self.interactions_to_book_data_map[csv_id] = book_id
                self.book_data_to_interactions_map[book_id] = csv_id

        # Initializes vectorizer for use in search function
        self.vectorizer = TfidfVectorizer()
        self.tfidf = self.vectorizer.fit_transform(self.titles["norm_titles"])

        self.liked_books: list[str] = []
        self.similar_users: set[int] = set()
        
        # Sets polars config for display purposes
        pl.Config.set_tbl_hide_dataframe_shape(True)
        pl.Config.set_tbl_hide_column_data_types(True) 
        pl.Config.set_fmt_str_lengths(100)

    # Finds book titles using term frequency-inverse document frequency
    def search(self, input: str) -> pl.DataFrame:
        norm_input = re.sub("[^a-z0-9 ]", "", input.lower())
        input_vector = self.vectorizer.transform([norm_input])
        similarity = cosine_similarity(input_vector, self.tfidf).flatten()
        # Finds the 50 most similar books
        indices = np.argpartition(similarity, -10)[-50:]
        results = self.titles[indices]
        # Sorts the top 50 results by ratings to find the most popular version of duplicate books
        results = results.sort("ratings", descending = True)
        results = results.select("book_id", "title", "url")
        return results.head(10)

    # Adds book to the list of liked books
    def add_liked_book(self, input: str) -> None:
        if input in self.liked_books:
            print("Book is already selected")
            return
        
        # Does not allow input that cannot be an integer
        try:
            tmp = int(input)
        except ValueError:
            print("Error: input is not a book id number")
            return
        
        # Does not allow inputs lower than the first book id
        if tmp > 0:
            self.liked_books.append(input)
    
    # Removes a single liked book based on its book id
    def remove_liked_book(self, input: str) -> None:
        if input in self.liked_books:
            self.liked_books.remove(input)
        else:
            print("This value does not exist in the list")

    # Find users who liked books from the list of liked books
    def find_similar_users(self) -> None:
        # Prevents the function from executing if no books were selected
        if not self.liked_books:
            print("Error: no books entered")
            return
        
        # Converts the book ids for use with interactions data
        liked_books_mapped = []
        for book in self.liked_books:
            liked_books_mapped.append(self.book_data_to_interactions_map[book])
            
        # Creates a dataframe of interactions containing liked books
        filtered_interactions_df = self.interactions_df.filter(pl.col("csv_id").is_in(liked_books_mapped))

        # Creates a set of similar users based on the list of liked books
        for row in filtered_interactions_df.iter_rows():
            user_id = row[0]
            
            # Prevents the function from trying to enter users multiple times
            if user_id in self.similar_users:
                continue
            
            rating = row[2]
            
            # Adds users that gave positive reviews for liked books
            if rating >= 4:
                self.similar_users.add(user_id)
    
    # Finds book recommendations
    def find_recs(self) -> pl.DataFrame:
        # Prevents the function from executing if there are no similar users (most likely due to not entering any books)
        if len(self.similar_users) == 0:
            print("Error: no similar users found")
            return pl.DataFrame()
        
        # Creates a dataframe of all interactions with similar users, selects books with positive ratings
        recs_df = self.interactions_df.filter(pl.col("user_id").is_in(self.similar_users) & pl.col("rating") > 0)
        recs_df = recs_df.filter(recs_df["rating"] >= 3)
        
        recs = recs_df["csv_id"].value_counts()
        recs.columns = ["book_id", "book_count"]
        
        recs = recs.filter(recs["book_count"] > 25)
        
        # Converts book ids from interactions data to books data
        book_id = []
        for row in recs.iter_rows():
            book_id.append(self.interactions_to_book_data_map[row[0]])
        
        book_id_series = pl.Series("book_id", book_id)
        recs = recs.with_columns(book_id_series)
        
        # Combines data with titles
        recs = recs.join(self.titles, how = "inner", on = "book_id")
        
        # Gives book a score that prioritizes books specifically popular among similar users
        recs = recs.with_columns((recs["book_count"] * (recs["book_count"] / recs["ratings"])).alias("score"))
        
        # Sorts books by score and filters out liked books
        filtered_recs = recs.sort("score", descending = True)
        filtered_recs = filtered_recs.filter(~pl.col("book_id").is_in(self.liked_books))
        filtered_recs = filtered_recs.select("book_id", "title", "url")
        
        return filtered_recs.head(10)

    # Reset the list of liked books
    def reset_liked_books(self) -> None:
        self.liked_books = []
    
    # Resets recommendation variables
    def reset_recs(self) -> None:
        self.liked_books = []
        self.similar_users = set()

def main():
    # Sets up directory for data files, change the directory argument based on file location
    parser = argparse.ArgumentParser(
                        prog="book_rec.py",
                        description="Recommends books based on Goodreads book ids")
    parser.add_argument("-d", "--directory", type=Path, default = ".")
    args = parser.parse_args()

    print("initializing")
    my_book_recommender = BookRecommender("goodreads_books.json.gz", "book_id_map.csv", "goodreads_interactions.csv", args.directory)
    while True:
        while True:
            val = input("Enter the title of a book you like: ")
            print(my_book_recommender.search(val))
            val = input("Enter the book id of the book you like or enter 0 to continue without selecting a book: ")
            my_book_recommender.add_liked_book(val)
            val = input("Would you like to search for another book? Enter 'Y' for yes or 'N' for no: ")
            if val.lower() == "n":
                break
        print("searching for users")
        my_book_recommender.find_similar_users()
        print("searching for recommendations")
        print(my_book_recommender.find_recs())
        
        val = input("Would you liked to search for more recommendations? Enter 'Y' for yes or 'N' for no: ")
        if val.lower() == "n":
            break
        val = input("Would you like to clear your list of liked books? Enter 'Y' for yes or 'N' for no: ")
        if val.lower() == "y":
            my_book_recommender.reset_recs()
        else:
            while True:
                val = input("Enter a book id from the top recommendations to add it to your list or enter 0 to return to searching: ")
                if (val == "0"):
                    break
                my_book_recommender.add_liked_book(val)

if __name__ == "__main__":
    main()
