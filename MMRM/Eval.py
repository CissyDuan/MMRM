import torch
from tqdm import tqdm
from transformers import AutoTokenizer#, AutoModel,RobertaForMaskedLM
#tokenizer = AutoTokenizer.from_pretrained("ethanyt/guwenbert-large")
tokenizer = AutoTokenizer.from_pretrained("./data")
from Main_cl import *
def eval_sp(model, dataloader):
    model.eval()
    topk = [20, 10, 5, 1]

    # block=1
    accs = []
    mrrs = []
    topks = [[], [], [], []]
    for batch in dataloader:
        mask_pic = batch.mask_pic
        block = len(mask_pic[0])
        break
    print(block)

    for i in range(block):
        num_correct = 0
        num_total = 0
        topk_sum = [0, 0, 0, 0]
        rr_sum = 0

        for batch in tqdm(dataloader):
            if len(batch.tgt_ids) != 0:

                batch_i = copy.deepcopy(batch)
                batch_i.mask_pic = [e[i] for e in batch_i.mask_pic]
                output = model.decode(batch_i)  # .tolist()
                pred = output.max(dim=1)[1]  # .unsqueeze(1)
                tgt = batch.tgt_ids
                tgt = torch.stack(tgt, dim=0).cuda().squeeze()
                correct = pred.data.eq(tgt.data).sum()
                num_correct += correct.data
                total = tgt.data.ne(1).sum()
                num_total += total.data

                output_sort, output_sortidx = torch.sort(output, dim=1, descending=True)
                mark = output_sortidx[:, :100].eq(tgt.unsqueeze(-1))
                idx = torch.arange(1, 101).cuda()
                r = torch.mul(idx, mark).sum(1)
                r = torch.where(r == 0, 10000.0, r)
                rr = 1 / r
                rr = torch.where(rr == 1 / 10000.0, 0, rr)
                rr = rr.sum().data
                rr_sum += rr

                for j in range(len(topk)):
                    top = topk[j]
                    mark_j = mark[:, :top]
                    topk_sum[j] += mark_j.sum().data

        acc = num_correct.float() / num_total.float()
        acc = acc.data.item()
        mrr = rr_sum.float() / num_total.float()
        mrr = mrr.data.item()
        topacc = []
        for k in topk_sum:
            topacci = k.float() / num_total.float()
            topacc.append(topacci.data.item())
        row = [i + 1, acc, mrr]
        for k in range(len(topk)):
            row.append(topk[k])
            row.append(topacc[k])

        logging_csv(row)
        logging('acc{},block{}\n'.format(acc, i))
        accs.append(acc)
        mrrs.append(mrr)
        for k in range(len(topacc)):
            topks[k].append(topacc[k])

    row = ['avg', sum(accs) / len(accs), sum(mrrs) / len(mrrs)]
    for k in topks:
        row.append(sum(k) / len(k))
    # logging_csv(row)
    print(row)
    # return sum(accs)/len(accs)
    return row


def eval_multi(model, dataloader_test):
    model.eval()
    step = 30
    result_sum = [0, 0, 0, 0, 0, 0]
    result_max = [0, 0, 0, 0, 0, 0]
    result_min = [100, 100, 100, 100, 100, 100]

    for i in range(step):
        row = eval_sp(model, dataloader_test)
        row = row[1:]
        for j in range(len(row)):
            result_sum[j] += row[j]
            if row[j] > result_max[j]:
                result_max[j] = row[j]
            if row[j] < result_min[j]:
                result_min[j] = row[j]

    avg = ['final_avg'] + [x / step for x in result_sum]
    maxv = ['max'] + result_max
    minv = ['min'] + result_min
    logging_csv(avg)
    logging_csv(maxv)
    logging_csv(minv)

def eval_sp_n(model, dataloader):
    model.eval()
    topk = [20, 10, 5, 1]

    num_correct = 0
    num_total = 0
    topk_sum = [0, 0, 0, 0]
    rr_sum = 0

    for batch in tqdm(dataloader):
        if len(batch.tgt_ids) != 0:

            output = model.decode(batch)  # .tolist()
            pred = output.max(dim=1)[1]  # .unsqueeze(1)
            tgt = batch.tgt_ids
            tgt = torch.stack(tgt, dim=0).cuda().squeeze()
            correct = pred.data.eq(tgt.data).sum()
            num_correct += correct.data
            total = tgt.data.ne(1).sum()
            num_total += total.data

            output_sort, output_sortidx = torch.sort(output, dim=1, descending=True)
            mark = output_sortidx[:, :100].eq(tgt.unsqueeze(-1))
            idx = torch.arange(1, 101).cuda()
            r = torch.mul(idx, mark).sum(1)
            r = torch.where(r == 0, 10000.0, r)
            rr = 1 / r
            rr = torch.where(rr == 1 / 10000.0, 0, rr)
            rr = rr.sum().data
            rr_sum += rr

            for j in range(len(topk)):
                top = topk[j]
                mark_j = mark[:, :top]
                topk_sum[j] += mark_j.sum().data

    acc = num_correct.float() / num_total.float()
    acc = acc.data.item()
    mrr = rr_sum.float() / num_total.float()
    mrr = mrr.data.item()
    topacc = []
    for k in topk_sum:
        topacci = k.float() / num_total.float()
        topacc.append(topacci.data.item())
    row = ['avg',acc, mrr]
    for k in range(len(topk)):
        #row.append(topk[k])
        row.append(topacc[k])

    logging_csv(row)
    print(row)

    return row

def eval_multi_n(model, dataloader_test):
    model.eval()
    step = 10
    result_sum = [0, 0, 0, 0, 0, 0]
    result_max = [0, 0, 0, 0, 0, 0]
    result_min = [100, 100, 100, 100, 100, 100]

    for i in range(step):
        row = eval_sp_n(model, dataloader_test)
        row = row[1:]
        for j in range(len(row)):
            result_sum[j] += row[j]
            if row[j] > result_max[j]:
                result_max[j] = row[j]
            if row[j] < result_min[j]:
                result_min[j] = row[j]

    avg = ['final_avg'] + [x / step for x in result_sum]
    maxv = ['max'] + result_max
    minv = ['min'] + result_min
    logging_csv(avg)
    logging_csv(maxv)
    logging_csv(minv)

def eval_pic(model, dataloader):
    model.eval()

    multi_ref = []
    candidate = []
    sources = []
    topks = []

    num_correct = 0
    num_total = 0
    count = 1

    for batch in tqdm(dataloader):

        if len(batch.tgt_ids) != 0:
            source = batch.s_ids_mask
            ref = batch.tgt_ids

            output, output_pic = model.decode_pic(batch)  # .tolist()

            pred = output.max(dim=1)[1]

            output_sort, output_sortidx = torch.sort(output, dim=1, descending=True)
            topk = output_sortidx[:, :10]

            tgt = batch.tgt_ids
            tgt = torch.stack(tgt, dim=0).cuda().squeeze()
            correct = pred.data.eq(tgt.data).sum()
            num_correct += correct.data
            total = tgt.data.ne(1).sum()
            num_total += total.data
            pred = torch.unbind(pred.unsqueeze(1))

            source_len = torch.stack(batch.s_mask_pad).sum(1).squeeze()
            source_len = source_len.tolist()

            source = [tokenizer.convert_ids_to_tokens(s[:l], skip_special_tokens=False) for s, l in
                      zip(source, source_len)]
            cand = [tokenizer.convert_ids_to_tokens(o, skip_special_tokens=False) for o in pred]
            ref = [tokenizer.convert_ids_to_tokens(r, skip_special_tokens=False) for r in ref]
            topk_cand = [tokenizer.convert_ids_to_tokens(t, skip_special_tokens=False) for t in topk]

            multi_ref += ref
            candidate += cand
            sources += source
            topks += topk_cand

            output_pic = [p.cpu() for p in output_pic]
            pic_mask = batch.mask_pic
            pic_mask = [p.cpu() for p in pic_mask]
            pic_ori = batch.pic
            pic_ori = [p.cpu() for p in pic_ori]

            show = [pic_ori, pic_mask, output_pic]
            for i in range(len(show)):
                p = show[i]
                image_show = image_combine(p, 10)
                image_show = trans_pil(image_show)
                image_show.save(log_path + "pic/pic_" + str(count) + '_' + str(i) + ".jpg")
            count += 1

    acc = num_correct.float() / num_total.float()
    acc = acc.data.item()

    logging_csv([99, 99, acc])
    logging('acc{}\n'.format(acc))

    print_list = [sources, candidate, multi_ref, topks]
    utils.write_result_to_file(print_list, log_path)
    model.train()

    return acc


def eval(model, dataloader, epoch, updates):
    model.eval()

    multi_ref = []
    candidate = []
    sources = []
    num_correct = 0
    num_total = 0

    for batch in tqdm(dataloader):
        if len(batch.tgt_ids) != 0:
            source = batch.s_ids_mask
            ref = batch.tgt_ids

            output = model.decode(batch)  # .tolist()
            pred = output.max(dim=1)[1]  # .unsqueeze(1)
            tgt = batch.tgt_ids
            tgt = torch.stack(tgt, dim=0).cuda().squeeze()
            correct = pred.data.eq(tgt.data).sum()
            num_correct += correct.data
            total = tgt.data.ne(1).sum()
            num_total += total.data
            pred = torch.unbind(pred.unsqueeze(1))
            source_len = torch.stack(batch.s_mask_pad).sum(1).squeeze()
            source_len = source_len.tolist()

            source = [tokenizer.convert_ids_to_tokens(s[:l], skip_special_tokens=False) for s, l in
                      zip(source, source_len)]
            # source = [tokenizer.convert_ids_to_tokens(s, skip_special_tokens=False) for s in source]
            cand = [tokenizer.convert_ids_to_tokens(o, skip_special_tokens=False) for o in pred]
            ref = [tokenizer.convert_ids_to_tokens(r, skip_special_tokens=False) for r in ref]

            multi_ref += ref
            candidate += cand
            sources += source

    acc = num_correct.float() / num_total.float()
    acc = acc.data.item()

    logging_csv([epoch, updates, acc])
    logging('acc{}\n'.format(acc))

    print_list = [sources, candidate, multi_ref]
    utils.write_result_to_file(print_list, log_path)

    model.train()

    return acc


def eval_real_lm(model, dataloader):
    model.eval()
    candidate = []
    sources = []
    topks = []
    ranks = []

    for batch in tqdm(dataloader):
        source = batch.s_ids_mask
        output = model.decode(batch)  # .tolist()

        pred = output.max(dim=1)[1]
        tgt = batch.tgt_ids
        tgt = torch.stack(tgt, dim=0).cuda()

        output_sort, output_sortidx = torch.sort(output, dim=1, descending=True)
        # print(tgt.size())
        # print(output_sortidx.size())

        mark = output_sortidx[:, :100].eq(tgt.unsqueeze(-1))
        idx = torch.arange(1, 101).cuda()
        r = torch.mul(idx, mark).sum(1)
        r = torch.where(r == 0, 10000.0, r)

        r = r.tolist()
        r = [str(ri) for ri in r]

        topk = output_sortidx[:, :20]

        pred = torch.unbind(pred.unsqueeze(1))

        source_len = torch.stack(batch.s_mask_pad).sum(1).squeeze()
        source_len = source_len.tolist()

        source = [tokenizer.convert_ids_to_tokens(s[:l], skip_special_tokens=False) for s, l in zip(source, source_len)]
        cand = [tokenizer.convert_ids_to_tokens(o, skip_special_tokens=False) for o in pred]
        topk_cand = [tokenizer.convert_ids_to_tokens(t, skip_special_tokens=False) for t in topk]

        candidate += cand
        sources += source
        topks += topk_cand
        ranks += r

    print_list = [sources, candidate, topks, ranks]
    utils.write_result_to_file(print_list, log_path)


def eval_real(model, dataloader):
    model.eval()
    candidate = []
    sources = []
    topks = []
    ranks = []

    for batch in tqdm(dataloader):

        source = batch.s_ids_mask
        output, output_pic = model.decode_pic(batch)  # .tolist()

        pred = output.max(dim=1)[1]
        tgt = batch.tgt_ids
        tgt = torch.stack(tgt, dim=0).cuda()

        output_sort, output_sortidx = torch.sort(output, dim=1, descending=True)
        # print(tgt.size())
        # print(output_sortidx.size())

        mark = output_sortidx[:, :100].eq(tgt.unsqueeze(-1))
        idx = torch.arange(1, 101).cuda()
        r = torch.mul(idx, mark).sum(1)
        r = torch.where(r == 0, 10000.0, r)

        r = r.tolist()
        r = [str(ri) for ri in r]

        topk = output_sortidx[:, :20]

        pred = torch.unbind(pred.unsqueeze(1))

        source_len = torch.stack(batch.s_mask_pad).sum(1).squeeze()
        source_len = source_len.tolist()

        source = [tokenizer.convert_ids_to_tokens(s[:l], skip_special_tokens=False) for s, l in zip(source, source_len)]
        cand = [tokenizer.convert_ids_to_tokens(o, skip_special_tokens=False) for o in pred]
        topk_cand = [tokenizer.convert_ids_to_tokens(t, skip_special_tokens=False) for t in topk]

        candidate += cand
        sources += source
        topks += topk_cand
        ranks += r

        output_pic = [p.cpu() for p in output_pic]
        show = [output_pic]
        for i in range(len(show)):
            p = show[i]
            image_show = image_combine(p, 10)
            image_show = trans_pil(image_show)
            image_show.save(log_path + "/pic_" + '_' + str(i) + ".jpg")

    print_list = [sources, candidate, topks, ranks]
    utils.write_result_to_file(print_list, log_path)
