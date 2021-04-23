# pip install transformers pytorch-lightning
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from model import LitModel
from datamodule import LitDataModule



def get_file_caption(data_dir, indices):        
    files = []
    captions = []
    for i in indices:
        files.append(os.path.join(data_dir, 'images/') + str(i) + '.jpg')
        text = open(os.path.join(data_dir, 'captions/') + str(i) + '.txt')
        caption = text.read()
        caption = caption.split('\n')[:-1]
        captions.append(caption)
        text.close()
    return files, captions


def inv_tfm(img, mean, std):
    for i in range(len(mean)):
        img[i,:,:] = img[i,:,:] * std[i] + mean[i]
    return img



def main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--data_dir', default='/workspace/data/face/CelebA_HQ_multi_modal/', type=str)
    parser.add_argument('--text_model', default='distilbert-base-multilingual-cased', type=str)
    parser.add_argument('--img_size', default=112, type=int)
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument('--transformer_embed_dim', default=768, type=int)
    parser.add_argument('--max_len', default=32, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--milestones', default=[120,150,180], type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    pl.seed_everything(args.seed)


    # ------------
    # model
    # ------------
    model = LitModel(args)
    # model = LitModel(args).load_from_checkpoint(args.checkpoint)
    ckpt = torch.load(args.checkpoint)
    model.load_state_dict(ckpt['state_dict'])


    tokenizer = Tokenizer(AutoTokenizer.from_pretrained(args.text_model), args.max_len)

    # Data
    infile = open(os.path.join(args.data_dir, 'filenames_test.pickle'),'rb')
    valid_indices = pickle.load(infile)
    infile.close()
        

    mean = [0.5, 0.5, 0.5]
    std = [0.25, 0.25, 0.25]

    transform_train = A.Compose([
        A.Resize(args.img_size, args.img_size),            
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()], p=1.)



    target_tfm = lambda x: tokenizer(random.choice(x))

    # valid dataset
    files, captions = get_file_caption(args.data_dir, valid_indices)
    valid_data = AlbumentationsDataset(
        file_paths=files,
        captions=captions,
        transform=transform_train,
        target_tfm=target_tfm
    )



    o_dir = 'results/'
    shutil.rmtree(o_dir, ignore_errors=True)
    os.makedirs(o_dir)


    print('Number of samples: ', len(valid_data))
    img, caption = valid_data[3] # load 4th sample

    print("Image Size: ", img.size())
    print(tokenizer.decode(caption))


    img = inv_tfm(img, mean, std) 




    plt.imshow(np.rot90(img.transpose(0, 2), 3))
    plt.title(tokenizer.decode(caption)[0])
    # plt.show()
    plt.savefig(o_dir+'sample_result.png')


    # valid_dl = DataLoader(dataset=valid_data, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=False)
    valid_dl = DataLoader(dataset=valid_data, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True)





    # Results
    # I will compare the text embeddings of the first batch (in the validation set) to all the images of the validation set by taking the dot product between them
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    _, y = next(iter(valid_dl))
    caption_encoder = model.caption_encoder
    caption_encoder.eval()
    caption_encoder = caption_encoder.to(device)

    with torch.no_grad():
        text_dev = {k: v.to(device) for k, v in y.items()}
        caption_embed = caption_encoder(text_dev)


    vision_encoder = model.vision_encoder
    vision_encoder.eval()
    vision_encoder = vision_encoder.to(device)

    image_embeds = []
    with torch.no_grad():
        for x, _ in tqdm(valid_dl):
            image_embeds.append(vision_encoder(x.to(device)))
    image_embed = torch.cat(image_embeds)


    similarity = caption_embed @ image_embed.T
    val, closest = similarity.topk(5, dim=-1)
    similarity.shape


    def draw_result(idx, similarity):
        text = dict()
        text['input_ids'] = y['input_ids'][idx]
        text['attention_mask'] = y['attention_mask'][idx]

        print('input:', tokenizer.decode(text))
        fig, ax = plt.subplots(1, 6, figsize=(12, 5))
        fig.suptitle(tokenizer.decode(text)[0], fontsize=12)


        val, closest = similarity.topk(5, dim=-1)

        for i in range(5):
            img, txt = valid_data[closest[idx, i]]
            print('similar txt:', i,  tokenizer.decode(txt))

            img = inv_tfm(img, mean, std) 
            ax[i].imshow(np.rot90(img.transpose(0, 2), 3))
            similarity_cap2img = val[idx, i].item()
            ax[i].set_title(f"{similarity_cap2img:.4f}")

        img, _ = valid_data[idx]
        img = inv_tfm(img, mean, std) 
        ax[-1].imshow(np.rot90(img.transpose(0, 2), 3))
        similarity_cap2img = similarity[idx, idx]
        ax[-1].set_title(f"True {similarity_cap2img:.4f}")
        plt.savefig(o_dir+'test_result_img_'+str(idx)+'.png')

        plt.hist(similarity[idx].cpu().numpy(), 100)
        plt.savefig(o_dir+'test_result_hist_'+str(idx)+'.png')


    for i in range(16):
        draw_result(i, similarity)


'''
    def draw_result_single_query(text, idx, similarity):
        fig, ax = plt.subplots(1, 6, figsize=(12, 5))
        fig.suptitle(text, fontsize=12)
        val, closest = similarity.topk(5, dim=-1)


        for i in range(5):
            img, txt = valid_data[closest[0, i]]
            print('similar txt:', i,  tokenizer.decode(txt))
            img = inv_tfm(img, mean, std) 
            ax[i].imshow(np.rot90(img.transpose(0, 2), 3))
            similarity_cap2img = val[0, i].item()
            ax[i].set_title(f"{similarity_cap2img:.4f}")

        img, y = valid_data[idx]
        img = inv_tfm(img, mean, std) 
        ax[-1].imshow(np.rot90(img.transpose(0, 2), 3))
        similarity_cap2img = similarity[0, idx]
        ax[-1].set_title(f"True {similarity_cap2img:.4f}")
        plt.savefig(o_dir+'single_query_result_img_'+str(idx)+'.png')


        # print(y)
        print(tokenizer.decode(y))

        plt.hist(similarity[0].cpu().numpy(), 100)
        plt.savefig(o_dir+'single_query_result_hist_'+str(idx)+'.png')

    text =  "A zebra standing up with it's head down and eating grass on the dirt ground."
    print('query text:', text)
    # text_dev = {k: v.to(device) for k, v in tokenizer(text).items()}
    text_dev = {k: v.unsqueeze(dim=0).to(device) for k, v in tokenizer(text).items()}
    with torch.no_grad():
        caption_embed_text = caption_encoder(text_dev)

    similarity_text = caption_embed_text @ image_embed.T

    draw_result_single_query(text, 4, similarity_text)

    # Guy and woman in glasses shake hands while exchanging gifts.
    text = "Guy and woman in glasses shake hands while exchanging gifts."
    print('query text:', text)
    text_dev = {k: v.unsqueeze(dim=0).to(device) for k, v in tokenizer(text).items()}
    with torch.no_grad():
        caption_embed_text = caption_encoder(text_dev)

    similarity_text = caption_embed_text @ image_embed.T

    draw_result_single_query(text, 5, similarity_text)



    text = "A shop filled with different kinds of clocks."
    print('query text:', text)
    text_dev = {k: v.unsqueeze(dim=0).to(device) for k, v in tokenizer(text).items()}
    with torch.no_grad():
        caption_embed_text = caption_encoder(text_dev)

    similarity_text = caption_embed_text @ image_embed.T

    draw_result_single_query(text, 6, similarity_text)


    # And lastly I check a single word version. Notice how the dog does kind of look like a bear. 
    # Maybe it's name is bear?
    text = "bear"
    text_dev = {k: v.unsqueeze(dim=0).to(device) for k, v in tokenizer(text).items()}
    with torch.no_grad():
        caption_embed_text = caption_encoder(text_dev)

    similarity_text = caption_embed_text @ image_embed.T

    draw_result_single_query(text, 7, similarity_text)

'''




if __name__ == '__main__':
    main()

