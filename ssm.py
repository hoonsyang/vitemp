 grad_all = 0
    for images, images_ID,  gt_cpu in tqdm(data_loader):    
        gt = gt_cpu.cuda()
        images = images.cuda()
        img_dct = dct.dct_2d(images)
        img_dct = V(img_dct, requires_grad = True)
        img_idct = dct.idct_2d(img_dct)

        output_ = model(img_idct)
        loss = F.cross_entropy(output, gt)
        loss.backward()
        grad = img_dct.grad.data
        grad = grad.mean(dim = 1).abs().sum(dim = 0).cpu().numpy()
        grad_all = grad_all + grad
        

    x = grad_all / 1000.0
    x = (x - x.min()) / (x.max() - x.min())
    g1 = sns.heatmap(x, cmap="rainbow")
    g1.set(yticklabels=[])  # remove the tick labels
    g1.set(ylabel=None)  # remove the axis label
    g1.set(xticklabels=[])  # remove the tick labels
    g1.set(xlabel=None)  # remove the axis label
    g1.tick_params(left=False)
    g1.tick_params(bottom=False)
    sns.despine(left=True, bottom=True)
    plt.show()
    plt.savefig("fig.png")