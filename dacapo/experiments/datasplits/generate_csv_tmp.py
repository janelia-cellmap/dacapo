import os

raw_folder = "/nrs/cellmap/data/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.n5"
raw_dataset = "em/fibsem-uint8/"

main_folder = "/nrs/cellmap/zouinkhim/data/tmp_data_v3/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.n5"
subfoder = "volumes/groundtruth/crop{crop}/labels/"

crops = [f[4:] for f in os.listdir(os.path.join(main_folder,"volumes/groundtruth/")) if f.startswith("crop") ]


# print(crops,len(crops))

# for c in crops:
#     print(c,os.listdir(os.path.join(main_folder,subfoder.format(crop=c))))

filtered_crops = [f for f in crops if os.path.exists(os.path.join(main_folder,subfoder.format(crop=f),"mito"))]

print(filtered_crops,len(filtered_crops))

from datasplit_generator import DataSplitGenerator, DatasetSpec, DatasetType

specs = []
for c in filtered_crops:
    specs.append(DatasetSpec(
        dataset_type= DatasetType.train,
        raw_container= raw_folder,
        raw_dataset= raw_dataset,
        gt_container= main_folder,
        gt_dataset= os.path.join(subfoder.format(crop=c),"mito"),
    ))
    specs.append(DatasetSpec(
        dataset_type= DatasetType.val,
        raw_container= raw_folder,
        raw_dataset= raw_dataset,
        gt_container= main_folder,
        gt_dataset= os.path.join(subfoder.format(crop=c),"mito"),
    ))

csv_path = "/groups/cellmap/cellmap/zouinkhim/dacapo_release/dacapo/dacapo/experiments/datasplits/test.csv"

DataSplitGenerator.generate_csv(specs, csv_path)


