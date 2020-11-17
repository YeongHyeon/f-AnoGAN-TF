import os, math

import numpy as np
import matplotlib.pyplot as plt

def make_dir(path):

    try: os.mkdir(path)
    except: pass

def gray2rgb(gray):

    rgb = np.ones((gray.shape[0], gray.shape[1], 3)).astype(np.float32)
    rgb[:, :, 0] = gray[:, :, 0]
    rgb[:, :, 1] = gray[:, :, 0]
    rgb[:, :, 2] = gray[:, :, 0]

    return rgb

def dat2canvas(data):

    numd = math.ceil(np.sqrt(data.shape[0]))
    [dn, dh, dw, dc] = data.shape
    canvas = np.ones((dh*numd, dw*numd, dc)).astype(np.float32)

    for y in range(numd):
        for x in range(numd):
            try: tmp = data[x+(y*numd)]
            except: pass
            else:
                canvas[(y*dh):(y*dh)+28, (x*dw):(x*dw)+28, :] = tmp
    if(dc == 1):
        canvas = gray2rgb(gray=canvas)

    return canvas

def save_img(contents, names=["", "", ""], savename=""):

    num_cont = len(contents)
    plt.figure(figsize=(5*num_cont+2, 5))

    for i in range(num_cont):
        plt.subplot(1,num_cont,i+1)
        plt.title(names[i])
        plt.imshow(dat2canvas(data=contents[i]))

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def boxplot(contents, savename=""):

    data, label = [], []
    for cidx, content in enumerate(contents):
        data.append(content)
        label.append("class-%d" %(cidx))

    plt.clf()
    fig, ax1 = plt.subplots()
    bp = ax1.boxplot(data, showfliers=True, whis=3)
    ax1.set_xticklabels(label, rotation=45)

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def random_noise(batch_size, zdim):

    return np.random.uniform(-1, 1, [batch_size, zdim]).astype(np.float32)

def training_wgan(neuralnet, dataset, epochs, batch_size, normalize=True):

    print("\nTraining WGAN to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    make_dir(path="training")
    result_list = ["from_noise"]
    if(neuralnet.zdim == 2): result_list.append("latent_walk")
    for result_name in result_list: make_dir(path=os.path.join("training", result_name))

    iteration = 0
    test_sq = 20
    test_size = test_sq**2
    for epoch in range(epochs):

        x_tr, y_tr, _ = dataset.next_train(batch_size=test_size, fix=True) # Initial batch
        z_tr = random_noise(x_tr.shape[0], neuralnet.zdim)
        step_dict = neuralnet.step(x=x_tr, z=z_tr, training=False, phase=0)
        x_fake = step_dict['x_fake']
        plt.imsave(os.path.join("training", "from_noise", "%08d.png" %(epoch)), dat2canvas(data=x_fake))

        if(neuralnet.zdim == 2):
            x_values = np.linspace(-3, 3, test_sq)
            y_values = np.linspace(-3, 3, test_sq)
            z_latents = None
            for y_loc, y_val in enumerate(y_values):
                for x_loc, x_val in enumerate(x_values):
                    z_latent = np.reshape(np.array([y_val, x_val]), (1, neuralnet.zdim))
                    if(z_latents is None): z_latents = z_latent
                    else: z_latents = np.append(z_latents, z_latent, axis=0)
            step_dict = neuralnet.step(x=x_tr, z=z_latents, training=False, phase=0)
            x_fake = step_dict['x_fake']
            z_tr = random_noise(x_tr.shape[0], neuralnet.zdim)
            plt.imsave(os.path.join("training", "latent_walk", "%08d.png" %(epoch)), dat2canvas(data=x_fake))


        while(True):
            x_tr, y_tr, terminator = dataset.next_train(batch_size)
            z_tr = random_noise(x_tr.shape[0], neuralnet.zdim)
            step_dict = neuralnet.step(x=x_tr, z=z_tr, iteration=iteration, training=True, phase=0)

            iteration += 1
            if(terminator): break

        print("Epoch [%d / %d] (%d iteration)  D:%.3f, G:%.3f" \
            %(epoch, epochs, iteration, step_dict['loss_d'], step_dict['loss_g']))
        neuralnet.save_parameter(model='model_checker', epoch=epoch)

    return iteration

def training_izi(neuralnet, dataset, epochs, batch_size, normalize=True, iteration=0):

    print("\nTraining izi to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    make_dir(path="training")
    result_list = ["restoration"]
    for result_name in result_list: make_dir(path=os.path.join("training", result_name))

    iteration = iteration
    test_sq = 20
    test_size = test_sq**2
    for epoch in range(epochs):

        x_tr, y_tr, _ = dataset.next_train(batch_size=test_size, fix=True) # Initial batch
        z_tr = random_noise(x_tr.shape[0], neuralnet.zdim)
        step_dict = neuralnet.step(x=x_tr, z=z_tr, training=False, phase=1)
        x_fake = step_dict['x_fake']
        save_img(contents=[x_tr, x_fake, (x_tr-x_fake)**2], \
            names=["Input\n(x)", "Restoration\n(x to x-hat)", "Difference"], \
            savename=os.path.join("training", "restoration", "%08d.png" %(epoch)))

        while(True):
            x_tr, y_tr, terminator = dataset.next_train(batch_size)
            z_tr = random_noise(x_tr.shape[0], neuralnet.zdim)
            step_dict = neuralnet.step(x=x_tr, z=z_tr, iteration=iteration, training=True, phase=1)

            iteration += 1
            if(terminator): break

        print("Epoch [%d / %d] (%d iteration)  E:%.3f" \
            %(epoch, epochs, iteration, step_dict['loss_e']))
        neuralnet.save_parameter(model='model_checker', epoch=epochs+epoch)

def test(neuralnet, dataset, batch_size):

    print("\nTest...")
    neuralnet.load_parameter(model='model_checker')

    make_dir(path="test")
    result_list = ["inbound", "outbound"]
    for result_name in result_list: make_dir(path=os.path.join("test", result_name))

    loss_list = []
    while(True):
        x_te, y_te, terminator = dataset.next_test(1)
        z_te = random_noise(1, neuralnet.zdim)
        step_dict = neuralnet.step(x=x_te, z=z_te, training=False, phase=1)
        x_fake, score_anomaly = \
            step_dict['x_fake'], np.sum(((x_te - step_dict['x_fake'])**2 + (1e-9))**(0.5))

        if(y_te[0] == 1):
            loss_list.append(score_anomaly)

        if(terminator): break

    loss_list = np.asarray(loss_list)
    loss_avg, loss_std = np.average(loss_list), np.std(loss_list)
    outbound = loss_avg + (loss_std * 3)
    print("Loss  avg: %.5f, std: %.5f" %(loss_avg, loss_std))
    print("Outlier boundary: %.5f" %(outbound))

    fcsv = open("test-summary.csv", "w")
    fcsv.write("class, loss, outlier\n")
    testnum = 0
    z_enc_tot, y_te_tot = None, None
    loss4box = [[], [], [], [], [], [], [], [], [], []]
    while(True):
        x_te, y_te, terminator = dataset.next_test(1)
        z_te = random_noise(1, neuralnet.zdim)
        step_dict = neuralnet.step(x=x_te, z=z_te, training=False, phase=1)
        x_fake, score_anomaly = \
            step_dict['x_fake'], np.sum(((x_te - step_dict['x_fake'])**2 + (1e-9))**(0.5))

        loss4box[y_te[0]].append(score_anomaly)

        outcheck = score_anomaly > outbound
        fcsv.write("%d, %.5f, %r\n" %(y_te, score_anomaly, outcheck))

        [h, w, c] = x_fake[0].shape
        canvas = np.ones((h, w*3, c), np.float32)
        canvas[:, :w, :] = x_te[0]
        canvas[:, w:w*2, :] = x_fake[0]
        canvas[:, w*2:, :] = (x_te[0]-x_fake[0])**2
        if(outcheck):
            plt.imsave(os.path.join("test", "outbound", "%08d-%08d.png" %(testnum, int(score_anomaly))), gray2rgb(gray=canvas))
        else:
            plt.imsave(os.path.join("test", "inbound", "%08d-%08d.png" %(testnum, int(score_anomaly))), gray2rgb(gray=canvas))

        testnum += 1

        if(terminator): break

    boxplot(contents=loss4box, savename="test-box.png")
