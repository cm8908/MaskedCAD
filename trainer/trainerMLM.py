from model.autoencoder import CADEncoder
from model.classifier import LinearClassifier
from .trainerCL import TrainerCL
from .loss import CadMLMLoss

class TrainerMLM(TrainerCL):
    def build_net(self, cfg):
        self.net = CADEncoder(cfg).cuda()
        self.classifier = LinearClassifier(cfg).cuda()
    
    def set_loss_function(self):
        self.mlm_loss = CadMLMLoss(self.cfg).cuda()
    
    def forward(self, data):
        command, args = data['command'].cuda(), data['args'].cuda()
        enc_output = self.net.forward(command, args)
        cmd_logits, args_logits = self.classifier(enc_output['representation'])
        
        # TEST!
        # len_command = command[0][command[0] != 3].shape[0]
        # print(command[0][:len_command])
        # print(args[0][:len_command])

        loss_dict = self.mlm_loss(cmd_logits, args_logits, data)
        return (enc_output, cmd_logits, args_logits), loss_dict
    
    def encode(self, data, is_batch=False):
        command, args = data['command'].cuda(), data['args'].cuda()
        if not is_batch:
            command, args = command.unsqueeze(0), args.unsqueeze(0)
        z = self.net.forward(command, args)['representation']
        return z