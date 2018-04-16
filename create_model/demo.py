import tifffile as tiff
from u_net_img import U_net

n_split = 4
crop_size = 288
patch_size = 224
get_size = 224
test_file_dir = '/home/yokoyang/PycharmProjects/untitled/896_val'
msk_file_dir = '/home/yokoyang/PycharmProjects/untitled/predict_img_result'
file_dir = test_file_dir

u_net = U_net(n_split, crop_size, patch_size, get_size, test_file_dir, msk_file_dir, file_dir)
all_class = ['countryside', 'playground', 'tree', 'road', 'building_yard', 'bare_land', 'water', 'general_building',
             'factory', 'shadow']
data_imageID_file = '/home/yokoyang/PycharmProjects/untitled/896_biaozhu/data_imageID.csv'

# for c in all_class:
#     u_net.train(c, data_imageID_file, 'unet1', epochs=100)

general_building = u_net.get_unet1()
general_building.load_weights('/home/yokoyang/PycharmProjects/untitled/model/general_building.hdf5')

tree_model = u_net.get_unet1()
tree_model.load_weights('/home/yokoyang/PycharmProjects/untitled/model/tree.hdf5')

water_model = u_net.get_unet1()
water_model.load_weights('/home/yokoyang/PycharmProjects/untitled/model/water.hdf5')

bare_land_model = u_net.get_unet1()
bare_land_model.load_weights('/home/yokoyang/PycharmProjects/untitled/model/bare_land.hdf5')

building_yard_model = u_net.get_unet1()
building_yard_model.load_weights('/home/yokoyang/PycharmProjects/untitled/model/building_yard.hdf5')

countryside_model = u_net.get_unet1()
countryside_model.load_weights('/home/yokoyang/PycharmProjects/untitled/model/countryside.hdf5')

factory_model = u_net.get_unet1()
factory_model.load_weights('/home/yokoyang/PycharmProjects/untitled/model/factory.hdf5')

playground_model = u_net.get_unet1()
playground_model.load_weights('/home/yokoyang/PycharmProjects/untitled/model/playground.hdf5')

road_model = u_net.get_unet1()
road_model.load_weights('/home/yokoyang/PycharmProjects/untitled/model/road.hdf5')

shadow_model = u_net.get_unet1()
shadow_model.load_weights('/home/yokoyang/PycharmProjects/untitled/model/shadow.hdf5')

file_id = '2_3.tif'
input_img = tiff.imread('/home/yokoyang/PycharmProjects/untitled/896_biaozhu/split-data/'+file_id)

result = u_net.predict_all_class(general_building, tree_model, water_model, bare_land_model,
                                 building_yard_model, countryside_model, factory_model, playground_model, road_model,
                                 shadow_model, input_img)
tiff.imsave('/home/yokoyang/PycharmProjects/untitled/check_result/'+file_id, result)
print(file_id)

#
# u_net.generate_all_files_result(general_building, tree_model, water_model, bare_land_model,
#                                 building_yard_model, countryside_model, factory_model, playground_model,
#                                 road_model, shadow_model, dir_name)
