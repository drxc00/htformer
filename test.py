from augment import DataAugmentor

if __name__ == '__main__':
    augmentor = DataAugmentor('data\\raw\\deadlifts', 'data\\augmented\\deadlifts')
    augmentor.augment("deadlift_2_rep_1.mp4")