from u_net_img import U_net

n_split = 4
crop_size = 224
patch_size = 192
get_size = 159
test_file_dir = '/home/yokoyang/PycharmProjects/untitled/896_val'
msk_file_dir = '/home/yokoyang/PycharmProjects/untitled/predict_img_result'
file_dir = '/home/yokoyang/PycharmProjects/untitled/896_biaozhu'

u_net = U_net(n_split, crop_size, patch_size, get_size, test_file_dir, msk_file_dir, file_dir)
all_class = ['countryside', 'playground', 'tree', 'road', 'building_yard', 'bare_land', 'water', 'general_building',
             'factory', 'shadow']
data_imageID_file = '/home/yokoyang/PycharmProjects/untitled/896_biaozhu/data_imageID.csv'

for c in all_class:
    u_net.train(c, data_imageID_file, 'unet1', epochs=100)

general_building = u_net.get_unet1()
general_building.load_weights('/home/yokoyang/PycharmProjects/untitled/model/general_building_1.hdf5')

tree_model = u_net.get_unet1()
tree_model.load_weights('/home/yokoyang/PycharmProjects/untitled/model/tree_1.hdf5')

water_model = u_net.get_unet1()
water_model.load_weights('/home/yokoyang/PycharmProjects/untitled/model/water_1.hdf5')

bare_land_model = u_net.get_unet1()
bare_land_model.load_weights('/home/yokoyang/PycharmProjects/untitled/model/bare_land_1.hdf5')

building_yard_model = u_net.get_unet1()
building_yard_model.load_weights('/home/yokoyang/PycharmProjects/untitled/model/building_yard_1.hdf5')

countryside_model = u_net.get_unet1()
countryside_model.load_weights('/home/yokoyang/PycharmProjects/untitled/model/countryside_1.hdf5')

factory_model = u_net.get_unet1()
factory_model.load_weights('/home/yokoyang/PycharmProjects/untitled/model/factory_1.hdf5')

playground_model = u_net.get_unet1()
playground_model.load_weights('/home/yokoyang/PycharmProjects/untitled/model/playground_1.hdf5')

road_model = u_net.get_unet1()
road_model.load_weights('/home/yokoyang/PycharmProjects/untitled/model/road_1.hdf5')

shadow_model = u_net.get_unet1()
shadow_model.load_weights('/home/yokoyang/PycharmProjects/untitled/model/shadow_1.hdf5')

dir_name = "shanghai2"
img_id = "0_0"
u_net.predict_all_class(general_building, tree_model, water_model, bare_land_model,
                        building_yard_model, countryside_model, factory_model, playground_model, road_model,
                        shadow_model, img_id, dir_name)
#
# u_net.generate_all_files_result(general_building, tree_model, water_model, bare_land_1_model,
#                                 building_yard_model, countryside_model, factory_model, playground_model,
#                                 road_model, shadow_model, dir_name)
