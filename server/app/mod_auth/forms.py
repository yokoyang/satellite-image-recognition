# Import Form and RecaptchaField (optional)
from flask_wtf import FlaskForm

# Import Form elements such as TextField and BooleanField (optional)
from wtforms import TextField, PasswordField

# Import Form validators
from wtforms.validators import Required, Email, EqualTo


# Define the login form (WTForms)

class LoginForm(FlaskForm):

    email = TextField('Email Address', [Email(),
                      Required(message='Forgot your email address?')])
    password = PasswordField('Password', [
                             Required(message='Must provide a password. ;-)')])
