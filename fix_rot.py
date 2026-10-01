from PIL import Image
im = Image.open("q11_oriented.jpg")
im.rotate(90, expand=True).save("q11_rot90.jpg")
