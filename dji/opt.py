import configargparse



def _get_opts():

    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    # parser.add_argument('--AT_images_path', default='/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan/origin/undistort',type=str, required=False)
    parser.add_argument('--original_images_path', default='/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan_lidar2/original_image/',type=str, required=False)
    parser.add_argument('--original_images_list_json_path', default='/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan_lidar2/original_image/images/survey//original_image_list.json',type=str, required=False)
    parser.add_argument('--infos_path', default='/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan_lidar2/original_image/AT/BlocksExchangeUndistortAT.xml',type=str, required=False)
    parser.add_argument('--dataset_path', default='/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan2',type=str, required=False)
    parser.add_argument('--resume', default=True, action='store_false')  # debug
    # parser.add_argument('--resume', default=False, action='store_true')  # run
    parser.add_argument('--num_val', type=int, default=10, help='Number of images to hold out in validation set')
    
    parser.add_argument('--las_path', default='/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan_lidar2/H2.las',type=str, required=False)
    parser.add_argument('--las_output_path', default='dji/process_las/output_2',type=str, required=False)
    parser.add_argument('--start', type=int, default=0, help='')
    parser.add_argument('--end', type=int, default=100000, help='')

    parser.add_argument('--output_path', default='dji/process_las/output_2',type=str, required=False)
    parser.add_argument('--down_scale', type=int, default=4, help='')
    parser.add_argument('--debug', type=eval, default=False, choices=[True, False],help='')


    ### 从质量报告中读取
    parser.add_argument('--fx', type=float, default=3697.407, help='')
    parser.add_argument('--cx', type=float, default=2712.971, help='')
    parser.add_argument('--cy', type=float, default=1809.624, help='')
    parser.add_argument('--k1', type=float, default=-0.009619294, help='')
    parser.add_argument('--k2', type=float, default=0.012821022, help='')
    parser.add_argument('--k3', type=float, default=-0.010474657, help='')
    parser.add_argument('--p1', type=float, default=-0.002084729, help='')
    parser.add_argument('--p2', type=float, default=-0.001407773, help='')

    return parser.parse_known_args()[0]
