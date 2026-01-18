from pypdf import PdfReader

reader = PdfReader("paper.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

with open("paper_content.txt", "w") as f:
    f.write(text)

print("Text extracted to paper_content.txt")
