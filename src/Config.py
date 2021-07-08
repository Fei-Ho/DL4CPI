ModelConfig = {'model': 'Embed_mlp',

               'Baseline': {'n_input': 2048 + 200,
                            'n_inner': 256,
                            'activation': 'leaky_relu'},

               'Embed_mlp': {'esm_dim': 768 * 7,
                             'mol_dim': 768 * 12,
                             # ['LayerNorm', 'ScaleNorm']
                             'norm': 'LayerNorm',
                             'n_inner': 2048,
                             'N': 3},

               'CPInet': {'esm_dim': 768,
                          'mol_dim': 1024,
                          'mol_feat_dim': 34,
                          'kernel_size': 7},
               }

DataConfig = {'DataType': 'AllChemBERT+AllESM',

              'Baseline': {'moleculeEmbedding': 'RDKFingerprint',
                           'proteinEmbedding': 'transformerCPI',
                           'UseMolFeat': False,
                           'data_path': '/lustre/S/wufandi/Project/CPI/CPI_Baseline/dataset/CPI2021JUN30',   # noqa
                           'embed_path': '../dataset/embeddings_Baseline',
                           'embed_avg': False,
                           },

              'RDK+ESM':    {'moleculeEmbedding': 'ChemBERT',
                             'proteinEmbedding': 'ESM',
                             'UseMolFeat': False,
                             'data_path': '/lustre/S/wufandi/Project/CPI/CPI_Baseline/dataset/CPI2021JUN30', # noqa
                             'embed_path': '../dataset/embeddings_RDKESM',                             # noqa
                             'embed_avg': True},

              'MAT+ESM':    {'moleculeEmbedding': 'MAT',
                             'proteinEmbedding': 'ESM',
                             'UseMolFeat': False,
                             'data_path': '/lustre/S/wufandi/Project/CPI/CPI_Baseline/dataset/CPI2021JUN30', # noqa
                             'embed_path': '../dataset/embeddings_MAT',
                             'embed_avg': True},

              'ChemBERT+ESM':    {'moleculeEmbedding': 'ChemBERT',
                                  'proteinEmbedding': 'ESM',
                                  'UseMolFeat': False,
                                  'data_path': '/lustre/S/wufandi/Project/CPI/CPI_Baseline/dataset/CPI2021JUL07', # noqa
                                  'embed_path': '../dataset/embeddingsBERTESM',
                                  'embed_avg': False},

              'Morgan+ESM': {'moleculeEmbedding': 'MorganFingerprint',
                             'proteinEmbedding': 'ESM',
                             'UseMolFeat': True,
                             'data_path':  '/mnt/storage_3/blai/CPI_data/dataset/CPI2021JUNE30', # noqa
                             'embed_path': '/mnt/storage_3/blai/CPI_data/dataset/embeddings_V3', # noqa
                             'embed_avg': True},

              'AllChemBERT+AllESM':    {'moleculeEmbedding': 'AllChemBERT',
                                        'proteinEmbedding': 'AllESM',
                                        'UseMolFeat': False,
                                        'data_path': '/lustre/S/wufandi/Project/CPI/CPI_Baseline/dataset/CPI2021JUL07', # noqa
                                        'embed_path': '../dataset/embeddingsAllBERT_AllESM',    # noqa
                                        'embed_avg': False},
              'ChemBERTout+ESM':    {'moleculeEmbedding': 'ChemBERTout',
                                     'proteinEmbedding': 'ESM',
                                     'UseMolFeat': False,
                                     'data_path': '/lustre/S/wufandi/Project/CPI/CPI_Baseline/dataset/CPI2021JUL07', # noqa
                                     'embed_path': '../dataset/embeddingsBERTout',    # noqa
                                     'embed_avg': False},
              }

TrainConfig = {'optim': {'optimizer': 'AdamW',
                         'lr': 1e-3,
                         'gamma': 0.99,
                         'weight_decay': 0.001,
                         'lr_scheduler_step_size': 5,
                         'lr_scheduler_gamma': 0.5},
               'MixPrec': False,
               'Epoch': 100,
               'Patience': 50,
               'BatchSize': 32,
               'ModelOut': '/lustre/S/wufandi/Project/CPI/CPI_Baseline/trained_models/Embed_mlp/ChemBERT_All.pt'}    # noqa
