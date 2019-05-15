import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.P = args.window; #10
        self.m1, self.m2, self.m3 = data.m1, data.m2, data.m3 # 3 dimensions of the 3D data
        self.hidR = args.hidRNN; #2
        self.hidC = args.hidCNN; #2
        self.hidS = args.hidSkip; #2
        self.Ck = args.CNN_kernel; #2
        self.skip = args.skip; #2
        self.pt = (self.P - self.Ck)/self.skip if self.skip > 0 else None
        self.hw = args.highway_window #2
        # self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m));
        
        # get the number of GPUs
        gpu_nums = torch.cuda.device_count()
        self.split_gpus = args.split_gpus
        gpu_id = 0

        # 3D case, model parallelism
        self.conv1 = nn.Conv3d(self.P, self.P * self.hidC * self.Ck, kernel_size = (self.m1, self.m2, self.m3));
        self.conv1.cuda(gpu_id)
        self.conv1_gpu = gpu_id # will be used in forward()
        if self.split_gpus and gpu_id < gpu_nums-1:
            gpu_id += 1

        self.GRU1 = nn.GRU(self.hidC, self.hidR);
        self.GRU1.cuda(gpu_id)
        self.GRU1_gpu = gpu_id
        self.dropout = nn.Dropout(p = 0.2);
        self.dropout.cuda(gpu_id)
        self.dropout_gpu = gpu_id
        if self.split_gpus and gpu_id < gpu_nums-1:
            gpu_id += 1
        
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS);
            self.linear1 = nn.Linear(self.hidR + int(self.skip) * self.hidS, self.m1*self.m2*self.m3);
            self.GRUskip.cuda(gpu_id)
            self.linear1.cuda(gpu_id)
            self.GRUskip_gpu = gpu_id
            self.linear1_gpu = gpu_id
            if self.split_gpus and gpu_id < gpu_nums-1:
                gpu_id += 1
        else:
            self.linear1 = nn.Linear(self.hidR, self.m1*self.m2*self.m3);
            self.linear1.cuda(gpu_id)
            self.linear1_gpu = gpu_id
            if self.split_gpus and gpu_id < gpu_nums-1:
                gpu_id += 1
            
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1);
            self.highway.cuda(gpu_id)
            self.highway_gpu = gpu_id
            if self.split_gpus and gpu_id < gpu_nums-1:
                gpu_id += 1
            
        self.output = None;
        if (args.output_fun == 'sigmoid'):
            self.output = torch.sigmoid;
            #self.output.cuda(gpu_id)
            #self.output_gpu = gpu_id
            #if self.split_gpus and gpu_id < gpu_nums-1:
            #    gpu_id += 1
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;
            #self.output.cuda(gpu_id)
            #self.output_gpu = gpu_id
            #if self.split_gpus and gpu_id < gpu_nums-1:
            #    gpu_id += 1
 
    def forward(self, x):
        batch_size = x.size(0);
        
        #CNN
        c = x.view(-1, self.P, self.m1, self.m2, self.m3);
        if self.use_cuda:
            print(self.conv1_gpu)
            c = c.cuda(self.conv1_gpu)
        c = F.relu(self.conv1(c));
        c = torch.squeeze(c.view(-1, self.hidC, self.Ck, self.P, 1));
        # print(c.size(), self.hidC, self.Ck, self.P)
        
        res1 = torch.zeros(c.size(0), self.hidC, self.P-self.Ck+1)
        if self.use_cuda:
            res1 = res1.cuda(self.conv1_gpu)
        for batch_index in range(c.size(0)):
            for channel in range(self.hidC):
                for begin in range(self.Ck):
                    res1[batch_index, channel, :] += c[batch_index, channel, begin, begin:begin+self.P-self.Ck+1] 
        c = res1
        if self.use_cuda:
            c = c.cuda(self.dropout_gpu)
        c = self.dropout(c);
        # print(c.shape) # [10, 3, 97]
        # c = torch.squeeze(c, 3);
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous();
        if self.use_cuda:
            r = r.cuda(self.GRU1_gpu)
        _, r = self.GRU1(r);
        if self.use_cuda:
            r = r.cuda(self.dropout_gpu)
        r = self.dropout(torch.squeeze(r,0));

        #skip-rnn
        if (self.skip > 0):
            #s = c[:,:, int(-self.pt * self.skip):].contiguous();
            s = c[:,:, -int(self.pt) * self.skip:].contiguous();
            # print(s.size(), batch_size, self.hidC, int(self.pt), self.skip)
            s = s.view(batch_size, self.hidC, int(self.pt), self.skip);
            s = s.permute(2,0,3,1).contiguous();
            s = s.view(int(self.pt), batch_size * self.skip, self.hidC);
            if self.use_cuda:
                s = s.cuda(self.GRUskip_gpu)
            _, s = self.GRUskip(s);
            s = s.view(batch_size, self.skip * self.hidS);
            if self.use_cuda:
                s = s.cuda(self.dropout_gpu)
            s = self.dropout(s);
            r = torch.cat((r,s),1);
        
        if self.use_cuda:
            r = r.cuda(self.linear1_gpu)
            #res = res.cuda(self.linear1_gpu)
        # print(r.get_device(), self.linear1_gpu) # the same
        res = self.linear1(r);
        #print(res.shape)
        
        #highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :, :, :];
            z = z.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.hw);
            if self.use_cuda:
                z = z.cuda(self.highway_gpu)
            z = self.highway(z);
            #print(z.shape)
            z = z.view(-1,self.m1*self.m2*self.m3);
            if self.use_cuda:
                res = res.cuda(self.highway_gpu)
            res = res + z;
            
        if (self.output):
            res = self.output(res);
            
        res = res.view(-1, self.m1, self.m2, self.m3)
        #print(res.shape) 
        return res;
