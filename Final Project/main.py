import article_search
import clustering
import rank_papers
import researcher_rec


def main():
    print("""
        Welcome to the researchers assistant! 
        The goal of this program, is to offer a selection of tools for navigating the academic world
    """)
    while True:
        option = int(input("""
            Choose an option:
            
            1.\tSearch for a topic
            2.\tCluster the article world
            3.\tFind rank of a researcher
            4.\tFind the top X researchers
            5.\tFind a research partner 
            6.\tExit
        """))

        if option == 1:
            query = input("Enter Query:\n")
            article_search.article_search(query)
        elif option == 2:
            method = input("Enter Method:\n")
            clusters = int(input("Enter number of clusters:\n"))
            clustering.cluster_best_articles(method, clusters)
            print("Your word clouds are under the clusters directory!")
        elif option == 3:
            name = input("Enter Researcher Name (Note that name might differ between articles):\n")
            rank_papers.researcher_ranking(name)
        elif option == 4:
            X = int(input("Enter number of researchers:\n"))
            rank_papers.top_rank_papers(X)
        elif option == 5:
            users_id = int(input("Enter your Semantic Scholar id"))
            researcher_rec.find_similar_researchers(users_id)
        else:
            print("Thanks for using the Research Assistant!")
            break


if __name__ == "__main__":
    main()
