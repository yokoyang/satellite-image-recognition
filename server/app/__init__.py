# Import flask and template operators
from flask import Flask, render_template, redirect

# Import SQLAlchemy
from flask_sqlalchemy import SQLAlchemy

from .mod_diag.u_net import U_net
from .mod_diag.ssd_net import SSD_net
from flask_cors import CORS

# Define the WSGI application object
app = Flask(__name__, instance_relative_config=True)
CORS(app)
ctx = app.app_context()
ctx.push()
# Configurations
app.config.from_object('config')
app.config.from_pyfile('config.py')

# app.config.from_envvar('APP_CONFIG_FILE')
# Define the database object which is imported
# by modules and controllers
# db = SQLAlchemy(app)

n_split = 4
crop_size = 288
patch_size = 160
get_size = 254
test_file_dir = ''
msk_file_dir = ''
u_net = U_net(n_split, crop_size, patch_size, get_size, test_file_dir, msk_file_dir)

general_building_model = u_net.get_unet1()
general_building_model.load_weights('app/mod_diag/weight/u_net/general_building.hdf5')

tree_model = u_net.get_unet1()
tree_model.load_weights('app/mod_diag/weight/u_net/tree.hdf5')

water_model = u_net.get_unet1()
water_model.load_weights('app/mod_diag/weight/u_net/water.hdf5')

bare_land_model = u_net.get_unet1()
bare_land_model.load_weights('app/mod_diag/weight/u_net/bare_land.hdf5')

building_yard_model = u_net.get_unet1()
building_yard_model.load_weights('app/mod_diag/weight/u_net/building_yard.hdf5')

countryside_model = u_net.get_unet1()
countryside_model.load_weights('app/mod_diag/weight/u_net/countryside.hdf5')

factory_model = u_net.get_unet1()
factory_model.load_weights('app/mod_diag/weight/u_net/factory.hdf5')

playground_model = u_net.get_unet1()
playground_model.load_weights('app/mod_diag/weight/u_net/playground.hdf5')

road_model = u_net.get_unet1()
road_model.load_weights('app/mod_diag/weight/u_net/road.hdf5')

shadow_model = u_net.get_unet1()
shadow_model.load_weights('app/mod_diag/weight/u_net/shadow.hdf5')

ssd_model = SSD_net('app/mod_diag/weight/intersection/ssd300.h5','app/mod_diag/weight/intersection/ssd300_weights.h5')
ssd_model.get_ssd()

class ModelClass:
    pass

dic_class = dict()
dic_class['water'] = water_model
dic_class['tree'] = tree_model
dic_class['playground'] = playground_model
dic_class['road'] = road_model
dic_class['building_yard'] = building_yard_model
dic_class['bare_land'] = bare_land_model
dic_class['general_building'] = general_building_model
dic_class['countryside'] = countryside_model
dic_class['factory'] = factory_model
dic_class['shadow'] = shadow_model
#
d = dict()
# # from 0 ~ 9
# # 0 mean on the top
d["road"] = 0
d["countryside"] = 1
d["factory"] =2

d["general_building"] = 3
d["playground"] = 4
d["water"] = 5
d["shadow"] = 6
d["bare_land"] = 7
d["building_yard"] = 8
d["tree"] = 9
#
#
set_classes = dict()
for key, value in d.items():
    model_class = ModelClass()
    model_class.order = value
    model_class.model = dic_class[key]
    set_classes[key] = model_class

# Sample HTTP error handling
@app.errorhandler(404)
def not_found(error):
    # return "Hello World"
    return render_template('404.html'), 404


@app.route('/')
def go_diagnose():
    return redirect('/diag')

# Import a module / component using its blueprint handler variable (mod_auth)
# from .mod_auth.controllers import mod_auth as auth_module
from .mod_diag.controllers import mod_diag as diag_module

# Register blueprint(s)
# app.register_blueprint(auth_module)
app.register_blueprint(diag_module)

# app.register_blueprint(xyz_module)
# ..

# Build the database:
# This will create the database file using SQLAlchemy
# db.create_all()
