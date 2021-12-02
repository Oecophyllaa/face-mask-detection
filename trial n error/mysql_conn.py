import mysql.connector

db = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="mask_db"
)

cursor = db.cursor()
sql = "INSERT INTO mask (mask_trig, deskripsi) VALUES (%s, %s)"
val = ("On", "Buzzer di nyalakan!")
cursor.execute(sql, val)
db.commit()

print("{} data ditambahkan".format(cursor.rowcount))


