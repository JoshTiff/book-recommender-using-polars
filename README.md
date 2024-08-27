# book-recommender-using-polars

This project is a program that uses a dataset of books and user ratings scraped from goodreads.com to give users book recommendations based on other books they like.

I developed this project because one of my friends was looking for book recommendations and the idea of a program that could create those recommendations seemed very interesting to explore. This project turned out to be very helpful in learning how to efficiently use data, as the datasets for the books and ratings turned out to be very large. 

In this project, a BookRecommender object is initialized and the user searches for books to add to a list of liked books. After the user creates their list of liked books, the program finds users who liked at least one book in the list and creates a set of similar users. The program then finds books that the similar users liked and uses them to provide book recommendations.

## Video Demo:
TO BE ADDED

## Features of the project include:
Effective search function: Term frequency-inverse document frequency prioritizes words that are less common in searches, so words like "the" will have less of an impact when searching. Book titles are also normalized to avoid distinctions based on capitalization or special characters. Additionally, the search function sorts the most similar results by rating so in the case of duplicate books, the most popular version will appear first.

Strong book recommendations: Recommendations use a formula to focus on books that are specifically popular with similar users to avoid books that are popular with everyone always begin recommended. Books are given a score from this equation to determine how likely it is that the user will like them and the final results are sorted by score to display the books they are most likely to enjoy.

Fast data manipulation: Although I originally began work on this project using Pandas, initial tests found Polars to complete the same tasks much more efficiently, so the project uses Polars for the best performance possible.

Reusability: After recommendations are given, users can add books from the recommendations to their liked books list or completely reset the list to get new recommendations without having to re-initialize the BookRecommender.

## How to install and run the project: 
Datasets were obtained from https://mengtingwan.github.io/data/goodreads.html

Install book_rec.py from here and the datasets goodreads_books.json.gz, goodreads_interactions.csv, and book_id_map.csv.<br />
Run the terminal command:<br />
python book_rec.py -d \<DIRECTORY\><br />
with <DIRECTORY> being replaced by the directory containing the data files. If they are in the same directory as book_rec.py, this argument can be omitted.
