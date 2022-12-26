class DissimNetPrior(nn.Module):
    def __init__(self, architecture='vgg16', semantic=True, pretrained=True, correlation=True, prior=False, spade='',
                 num_semantic_classes=19):
        super(DissimNetPrior, self).__init__()

        # get initialization parameters
        self.correlation = correlation
        self.spade = spade
        # self.semantic = False if spade else semantic
        self.semantic = semantic
        self.prior = prior

        # generate encoders
        if self.spade == 'encoder' or self.spade == 'both':
            self.vgg_encoder = VGGSPADE(pretrained=pretrained, label_nc=num_semantic_classes)
        else:
            self.vgg_encoder = VGGFeatures(architecture=architecture, pretrained=pretrained)

        if self.semantic:
            self.semantic_encoder = SemanticEncoder(architecture=architecture, in_channels=num_semantic_classes)
            self.prior_encoder = SemanticEncoder(architecture=architecture, in_channels=3, base_feature_size=64)

        # layers for decoder
        # all the 3x3 convolutions
        if correlation:
            self.conv1 = nn.Sequential(nn.Conv2d(513, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv12 = nn.Sequential(nn.Conv2d(513, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv3 = nn.Sequential(nn.Conv2d(385, 128, kernel_size=3, padding=1), nn.SELU())
            self.conv5 = nn.Sequential(nn.Conv2d(193, 64, kernel_size=3, padding=1), nn.SELU())

        else:
            self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv12 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv3 = nn.Sequential(nn.Conv2d(384, 128, kernel_size=3, padding=1), nn.SELU())
            self.conv5 = nn.Sequential(nn.Conv2d(192, 64, kernel_size=3, padding=1), nn.SELU())

        if self.spade == 'decoder' or self.spade == 'both':
            self.conv2 = SPADEDecoderLayer(nc=256, label_nc=num_semantic_classes)
            self.conv13 = SPADEDecoderLayer(nc=256, label_nc=num_semantic_classes)
            self.conv4 = SPADEDecoderLayer(nc=128, label_nc=num_semantic_classes)
            self.conv6 = SPADEDecoderLayer(nc=64, label_nc=num_semantic_classes)
        else:
            self.conv2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv13 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.SELU())
            self.conv4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.SELU())
            self.conv6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.SELU())

        # all the tranposed convolutions
        self.tconv1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0)
        self.tconv3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0)
        self.tconv2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0)

        # all the other 1x1 convolutions
        if self.semantic:
            self.conv7 = nn.Conv2d(1280, 512, kernel_size=1, padding=0)
            self.conv8 = nn.Conv2d(640, 256, kernel_size=1, padding=0)
            self.conv9 = nn.Conv2d(320, 128, kernel_size=1, padding=0)
            self.conv10 = nn.Conv2d(160, 64, kernel_size=1, padding=0)
            self.conv11 = nn.Conv2d(64, 2, kernel_size=1, padding=0)
        else:
            self.conv7 = nn.Conv2d(1024, 512, kernel_size=1, padding=0)
            self.conv8 = nn.Conv2d(512, 256, kernel_size=1, padding=0)
            self.conv9 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
            self.conv10 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
            self.conv11 = nn.Conv2d(64, 2, kernel_size=1, padding=0)

        # self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, original_img, synthesis_img, semantic_img, entropy, mae, distance, softmax_out=False):
        # get all the image encodings
        prior_img = torch.cat((entropy, mae, distance), dim=1)
        if self.spade == 'encoder' or self.spade == 'both':
            encoding_og = self.vgg_encoder(original_img, semantic_img)
            encoding_syn = self.vgg_encoder(synthesis_img, semantic_img)
        else:
            encoding_og = self.vgg_encoder(original_img)
            encoding_syn = self.vgg_encoder(synthesis_img)
    
        if self.semantic:
            encoding_sem = self.semantic_encoder(semantic_img)
            # concatenate the output of each encoder
            layer1_cat = torch.cat((encoding_og[0], encoding_syn[0], encoding_sem[0]), dim=1)
            layer2_cat = torch.cat((encoding_og[1], encoding_syn[1], encoding_sem[1]), dim=1)
            layer3_cat = torch.cat((encoding_og[2], encoding_syn[2], encoding_sem[2]), dim=1)
            layer4_cat = torch.cat((encoding_og[3], encoding_syn[3], encoding_sem[3]), dim=1)
        else:
            layer1_cat = torch.cat((encoding_og[0], encoding_syn[0]), dim=1)
            layer2_cat = torch.cat((encoding_og[1], encoding_syn[1]), dim=1)
            layer3_cat = torch.cat((encoding_og[2], encoding_syn[2]), dim=1)
            layer4_cat = torch.cat((encoding_og[3], encoding_syn[3]), dim=1)
    
        # use 1x1 convolutions to reduce dimensions of concatenations
        layer4_cat = self.conv7(layer4_cat)
        layer3_cat = self.conv8(layer3_cat)
        layer2_cat = self.conv9(layer2_cat)
        layer1_cat = self.conv10(layer1_cat)
        
        if self.prior:
            encoding_pior = self.prior_encoder(prior_img)
            layer1_cat = torch.mul(layer1_cat, encoding_pior[0])
            layer2_cat = torch.mul(layer2_cat, encoding_pior[1])
            layer3_cat = torch.mul(layer3_cat, encoding_pior[2])
            layer4_cat = torch.mul(layer4_cat, encoding_pior[3])
    
        if self.correlation:
            # get correlation for each layer (multiplication + 1x1 conv)
            corr1 = torch.sum(torch.mul(encoding_og[0], encoding_syn[0]), dim=1).unsqueeze(dim=1)
            corr2 = torch.sum(torch.mul(encoding_og[1], encoding_syn[1]), dim=1).unsqueeze(dim=1)
            corr3 = torch.sum(torch.mul(encoding_og[2], encoding_syn[2]), dim=1).unsqueeze(dim=1)
            corr4 = torch.sum(torch.mul(encoding_og[3], encoding_syn[3]), dim=1).unsqueeze(dim=1)
        
            # concatenate correlation layers
            layer4_cat = torch.cat((corr4, layer4_cat), dim=1)
            layer3_cat = torch.cat((corr3, layer3_cat), dim=1)
            layer2_cat = torch.cat((corr2, layer2_cat), dim=1)
            layer1_cat = torch.cat((corr1, layer1_cat), dim=1)
    
        # Run Decoder
        x = self.conv1(layer4_cat)
        if self.spade == 'decoder' or self.spade == 'both':
            x = self.conv2(x, semantic_img)
        else:
            x = self.conv2(x)
        x = self.tconv1(x)
    
        x = torch.cat((x, layer3_cat), dim=1)
        x = self.conv12(x)
        if self.spade == 'decoder' or self.spade == 'both':
            x = self.conv13(x, semantic_img)
        else:
            x = self.conv13(x)
        x = self.tconv3(x)
    
        x = torch.cat((x, layer2_cat), dim=1)
        x = self.conv3(x)
        if self.spade == 'decoder' or self.spade == 'both':
            x = self.conv4(x, semantic_img)
        else:
            x = self.conv4(x)
        x = self.tconv2(x)
    
        x = torch.cat((x, layer1_cat), dim=1)
        x = self.conv5(x)
        if self.spade == 'decoder' or self.spade == 'both':
            x = self.conv6(x, semantic_img)
        else:
            x = self.conv6(x)
        logits = self.conv11(x)
        return logits

