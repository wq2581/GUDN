import torch

from tqdm import tqdm
from model import ClassifyNet
from util import Accuracy, read_configuration, save
from dataset import get_train_data_loader, get_test_data_loader, get_label_num
from transformers import AdamW, get_scheduler, logging, BertTokenizer, BertModel


def init(config):
    logging.set_verbosity_error()
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    config['device'] = torch.device(f"cuda:{config['gpu_id']}")


def build_train_model(config):
    print("get label number")
    label_number = get_label_num(config['dataset'])

    print("build bert model")
    bert = BertModel.from_pretrained(f"./checkpoint/{config['dataset']}/{config['checkpoint_epoch']}" if config['use_checkpoint'] else config['bert_version']).to(config['device'])
    tokenizer = BertTokenizer.from_pretrained(f"./checkpoint/{config['dataset']}/{config['checkpoint_epoch']}" if config['use_checkpoint'] else config['bert_version'])

    print("build ClassifyNet model")
    classify_net = ClassifyNet(label_number=label_number, feature_layers=10, bert_hidden_size=bert.config.hidden_size).to(config['device'])
    if config['use_checkpoint']:
        classify_net.load_state_dict(torch.load(f"./checkpoint/{config['dataset']}/{config['checkpoint_epoch']}/student_model_file.bin"))

    print("build train data loader")
    train_data_loader = get_train_data_loader(config['dataset'], tokenizer=tokenizer, batch_size=config['train_batch_size'])

    print("build test data loader")
    test_data_loader = get_test_data_loader(config['dataset'], tokenizer=tokenizer, batch_size=config['train_batch_size'])

    print("build BCE loss")
    classify_loss_function = torch.nn.BCEWithLogitsLoss()

    print("build MSE loss")
    mse_loss_function = torch.nn.MSELoss()

    print("build optimizer")
    bert_train_parameters = [parameter for parameter in bert.parameters() if parameter.requires_grad]
    classify_net_train_parameters = [parameter for parameter in classify_net.parameters() if parameter.requires_grad]
    parameters = bert_train_parameters + classify_net_train_parameters
    optimizer = AdamW(parameters, lr=config['lr'])

    print("build lr_scheduler")
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=config['epochs'] * len(train_data_loader))

    return train_data_loader, test_data_loader, bert, classify_net, tokenizer, classify_loss_function, mse_loss_function, optimizer, lr_scheduler



def run_train_batch(config, data, bert, classify_net, classify_loss_function, mse_loss_function, optimizer, lr_scheduler):
    batch_text_input_ids, batch_text_padding_mask, batch_text_token_type_ids, \
    batch_label_input_ids, batch_label_padding_mask, batch_label_token_type_ids, \
    batch_label_one_hot = data

    batch_text_input_ids = batch_text_input_ids.to(config['device'])
    batch_text_padding_mask = batch_text_padding_mask.to(config['device'])
    batch_text_token_type_ids = batch_text_token_type_ids.to(config['device'])
    batch_label_input_ids = batch_label_input_ids.to(config['device'])
    batch_label_padding_mask = batch_label_padding_mask.to(config['device'])
    batch_label_token_type_ids = batch_label_token_type_ids.to(config['device'])
    batch_label_one_hot = batch_label_one_hot.to(config['device'])


    text_bert_out = bert(input_ids=batch_text_input_ids, attention_mask=batch_text_padding_mask,
                    token_type_ids=batch_text_token_type_ids, output_hidden_states=True)

    out = classify_net(text_bert_out)
    text_classify_loss = classify_loss_function(out, batch_label_one_hot)


    label_bert_out = bert(input_ids=batch_label_input_ids, attention_mask=batch_label_padding_mask,
                    token_type_ids=batch_label_token_type_ids, output_hidden_states=True)


    mse_loss = mse_loss_function(text_bert_out['pooler_output'], label_bert_out['pooler_output'])

    out = classify_net(label_bert_out)
    label_classify_loss = classify_loss_function(out, batch_label_one_hot)

    loss = text_classify_loss + mse_loss + label_classify_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()

    return text_classify_loss, mse_loss, label_classify_loss



def run_test_batch(config, data, bert, classify_net, loss_function, accuracy):
    batch_text_input_ids, batch_text_padding_mask, batch_text_token_type_ids, \
    batch_label_input_ids, batch_label_padding_mask, batch_label_token_type_ids, \
    batch_label_one_hot = data

    batch_text_input_ids = batch_text_input_ids.to(config['device'])
    batch_text_padding_mask = batch_text_padding_mask.to(config['device'])
    batch_text_token_type_ids = batch_text_token_type_ids.to(config['device'])
    batch_label_one_hot = batch_label_one_hot.to(config['device'])

    bert_out = bert(input_ids=batch_text_input_ids, attention_mask=batch_text_padding_mask,
                    token_type_ids=batch_text_token_type_ids, output_hidden_states=True)

    out = classify_net(bert_out)

    loss = loss_function(out, batch_label_one_hot)

    accuracy.calc(out, batch_label_one_hot)

    return loss


if __name__ == '__main__':
    config = read_configuration("./config.yaml")

    init(config)
    print(config)

    accuracy = Accuracy()

    train_data_loader, test_data_loader, bert, classify_net, tokenizer, loss_function, mse_loss_function, optimizer, lr_scheduler = build_train_model(config)

    save_acc1, save_acc3, save_acc5 = 0, 0, 0

    for epoch in range(config['epochs']):
        bert.train()
        classify_net.train()
        with tqdm(train_data_loader, ncols=200) as batch:
            for data in batch:
                text_classify_loss, mse_loss, label_classify_loss = run_train_batch(config, data, bert, classify_net, loss_function, mse_loss_function, optimizer, lr_scheduler)
                batch.set_description(f"train epoch:{epoch + 1}/{config['epochs']}")
                batch.set_postfix(text_classify_loss=text_classify_loss.item(), mse_loss=mse_loss.item(), label_classify_loss=label_classify_loss.item())
        with torch.no_grad():
            bert.eval()
            classify_net.eval()
            accuracy.reset_acc()
            with tqdm(test_data_loader, ncols=200) as batch:
                for data in batch:
                    _loss = run_test_batch(config, data, bert, classify_net, loss_function, accuracy)
                    batch.set_description(f"test epoch:{epoch + 1}/{config['epochs']}")
                    batch.set_postfix(loss=_loss.item(), p1=accuracy.get_acc1(), p3=accuracy.get_acc3(), p5=accuracy.get_acc5())