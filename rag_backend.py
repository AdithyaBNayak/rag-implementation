from langchain_community.document_loaders import PyPDFLoader

# Loading the pdf data
pdf_loader_obj = PyPDFLoader('https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf')
# print(pdf_loader_obj)
loaded_data = pdf_loader_obj.load_and_split()

no_of_pages = len(loaded_data)

print(f"Total No of Pages = {no_of_pages}\n\n\n")

for page_no, page_content in enumerate(loaded_data):
    print(f"Page {page_no}\n{page_content}\n\n\n")