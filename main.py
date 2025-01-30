# Created by Mohammed Jassim at 30/01/25
from project.rag_handler import rag_response


if __name__ == '__main__':
    print("Starting the rag experiments")

    while True:
        query = input("Query: ")

        if query:
            query_response = rag_response(query)

            print(query_response)

            exit_command = input("Do you want to quit (y/n): ")

            if exit_command.lower() != 'n':
                print('Exiting the program')
                break

        else:
            print('Exiting the program')
            break
