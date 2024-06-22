#cat wsgi.py
import sys
from app import server as application

sys.path.insert(0,"/var/www/html/Dash/")

if __name__ == "__main__":
    application.run()
